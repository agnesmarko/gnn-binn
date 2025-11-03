import pandas as pd
import torch
from torch import nn
from typing import Optional, List

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.binn_config import BinnConfig
from binn.src.model.pathway_network import PathwayNetwork
from binn.src.model.binn import BINN

def analyze_hierarchical_edge_index(edge_index, layer_dims):
    print(f"Total connections: {edge_index.shape[1]}")
    print(f"Total layers: {len(layer_dims)}")

    # Calculate layer boundaries
    layer_starts = [0]
    for dim in layer_dims:
        layer_starts.append(layer_starts[-1] + dim)

    # Print layer structure
    print("\nLayer Structure:")
    for i, dim in enumerate(layer_dims):
        if i == 0:
            layer_type = "Input"
        elif i == len(layer_dims) - 1:
            layer_type = "Output"
        else:
            layer_type = f"Hidden {i-1}"

        node_range = f"[{layer_starts[i]}-{layer_starts[i+1]-1}]"
        print(f"  {layer_type:>10}: {dim:>4} nodes {node_range}")

    # Count connections between each pair of layers
    layer_connections = {}
    for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        # Find which layer the source and target belong to
        source_layer = None
        target_layer = None

        for i in range(len(layer_starts) - 1):
            if layer_starts[i] <= s < layer_starts[i+1]:
                source_layer = i
            if layer_starts[i] <= t < layer_starts[i+1]:
                target_layer = i

        if source_layer is not None and target_layer is not None:
            key = (source_layer, target_layer)
            layer_connections[key] = layer_connections.get(key, 0) + 1

    # Print connections between layers
    print("\nConnections between layers:")
    output_layer_idx = len(layer_dims) - 1

    for (src, tgt), count in sorted(layer_connections.items()):
        skip_distance = tgt - src
        if skip_distance > 0:  # only show valid forward connections
            total_possible = layer_dims[src] * layer_dims[tgt]
            density = count / total_possible

            # Create readable layer names
            if src == 0:
                src_name = "Input"
            elif src == output_layer_idx:
                src_name = "Output"
            else:
                src_name = f"Hidden {src-1}"

            if tgt == 0:
                tgt_name = "Input"
            elif tgt == output_layer_idx:
                tgt_name = "Output"
            else:
                tgt_name = f"Hidden {tgt-1}"

            # Determine connection type
            if tgt == output_layer_idx:
                conn_type = "→ OUTPUT"
                # Verify only the last hidden layer connects to output
                if src != output_layer_idx - 1:
                    print(f"WARNING: {src_name} connects directly to output layer!")
            else:
                conn_type = "Adjacent" if skip_distance == 1 else f"Skip+{skip_distance}"

            print(f"  {src_name:>10} → {tgt_name:<10} ({conn_type:>9}): {count:>4} connections ({density:>6.2%} density)")

    # Additional validation
    print("\nValidation:")
    print(f"  Total nodes expected: {sum(layer_dims)}")
    print(f"  Total nodes in graph: {layer_starts[-1]}")

    # Check for any problematic connections
    problematic_connections = []
    for (src, tgt), count in layer_connections.items():
        if tgt <= src:  # backward or self-connections
            problematic_connections.append((src, tgt, count))

    if problematic_connections:
        print("Problematic connections found:")
        for src, tgt, count in problematic_connections:
            src_name = "Input" if src == 0 else f"Hidden {src-1}" if src < output_layer_idx else "Output"
            tgt_name = "Input" if tgt == 0 else f"Hidden {tgt-1}" if tgt < output_layer_idx else "Output"
            print(f"     {src_name} → {tgt_name}: {count} connections (backward/self)")
    else:
        print("All connections are forward-flowing")



def create_neural_network(config: BinnConfig, 
                          available_nodes_in_string: Optional[List[str]] = None, 
                          available_features_in_data: Optional[List[str]] = None
                        ) -> tuple[BINN, PathwayNetwork]:
    
    # read the protein input nodes
    protein_input_nodes = pd.read_csv(config.protein_input_nodes_path)

    pathway_net = PathwayNetwork(obo_path=config.obo_file_path,
                                protein_input_nodes=protein_input_nodes,
                                root_nodes_to_include=config.root_nodes_to_include,
                                gaf_is_preprocessed=True,
                                gaf_file_path=config.gaf_file_path,
                                preprocessed_gaf_file_path=config.preprocessed_gaf_file_path,
                                available_nodes_in_string=available_nodes_in_string,
                                available_features_in_data=available_features_in_data,
                                max_level=config.max_level,
                                verbose=True)


    # extract total nodes per level
    nodes_per_level = []
    for i in range(len(pathway_net.layer_indices) - 1):
        nodes_per_level.append(pathway_net.layer_indices[i+1] - pathway_net.layer_indices[i])


    # create input mapping (proteins are input)
    protein_level_idx = len(nodes_per_level) - 1  # last level is proteins
    protein_dim = nodes_per_level[protein_level_idx]

    # input dim depends on the data types specified in the config
    if config.data_type == 'combined':
        number_of_features_per_gene = 2  # CNV + mutation
    else:
        number_of_features_per_gene = 1  # only one data type
    input_dim = protein_dim * number_of_features_per_gene

    # hidden layers are protein layer + specific to general GO terms (excluding most general which will connect to output)
    # we need to reverse (since original order is general to specific)
    hidden_dims = [protein_dim]  # start with protein layer
    for i in reversed(range(0, protein_level_idx)):
        hidden_dims.append(nodes_per_level[i])


    # set output dim
    output_dim = config.output_dim

    # create neural network
    model = BINN(input_dim=input_dim,
                 hidden_dims=hidden_dims,
                 output_dim=output_dim,
                 norm = 'layer',
                 weight_init='custom')
    

    # create mapping between pathway indices and network indices
    pathway_to_network_idx = {}
    network_idx = input_dim  # start after input layer

    # map proteins to first hidden layer
    for i in range(pathway_net.layer_indices[protein_level_idx], pathway_net.layer_indices[protein_level_idx+1]):
        pathway_to_network_idx[i] = network_idx
        network_idx += 1

    # map GO term layers from specific to general 
    for level in reversed(range(0, protein_level_idx)):
        for i in range(pathway_net.layer_indices[level], pathway_net.layer_indices[level+1]):
            pathway_to_network_idx[i] = network_idx
            network_idx += 1

    # convert edge indices using the mapping (only for gene→GO and GO→GO connections)
    sources = []
    targets = []

    # add input→gene connections
    # each gene connects to exactly number_of_features_per_gene input nodes
    for protein_idx in range(protein_dim):
        protein_network_idx = input_dim + protein_idx

        feature_indices = [protein_dim * i + protein_idx for i in range(number_of_features_per_gene)]

        for feature_idx in feature_indices:
            sources.append(feature_idx)
            targets.append(protein_network_idx)

    # convert edge indices using the mapping (only for gene→GO and GO→GO connections)  
    for s, t in zip(pathway_net.edge_index[0], pathway_net.edge_index[1]):
        if s in pathway_to_network_idx and t in pathway_to_network_idx:
            sources.append(pathway_to_network_idx[s])
            targets.append(pathway_to_network_idx[t])


    # add connections from last hidden layer to output layer
    last_hidden_layer_start = sum(hidden_dims[:-1]) + input_dim
    last_hidden_layer_size = hidden_dims[-1]
    output_layer_start = sum(hidden_dims) + input_dim

    # connect every node in the last hidden layer to every output node
    for i in range(last_hidden_layer_size):
        source_idx = last_hidden_layer_start + i

        # connect to each output node 
        for j in range(output_dim):
            target_idx = output_layer_start + j

            sources.append(source_idx)
            targets.append(target_idx)


    edge_index = torch.tensor([sources, targets])



    # analyze the edge index
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    analyze_hierarchical_edge_index(edge_index, layer_dims)

    # max index in the edge index
    max_index = max(edge_index[0].max(), edge_index[1].max())
    print(f"Max index in edge index: {max_index.item()}")  

    # total nodes in the network
    total_nodes = pathway_net.graph.number_of_nodes()
    print(f"Total nodes in the network: {total_nodes}")

    # set connections
    model.set_connections(edge_index)

    return model, pathway_net


def create_pathway_network(config: BinnConfig, 
                          available_nodes_in_string: Optional[List[str]] = None, 
                          available_features_in_data: Optional[List[str]] = None
                        ) -> tuple[BINN, PathwayNetwork]:
    
    # read the protein input nodes
    protein_input_nodes = pd.read_csv(config.protein_input_nodes_path)

    pathway_net = PathwayNetwork(obo_path=config.obo_file_path,
                                protein_input_nodes=protein_input_nodes,
                                root_nodes_to_include=config.root_nodes_to_include,
                                gaf_is_preprocessed=True,
                                gaf_file_path=config.gaf_file_path,
                                preprocessed_gaf_file_path=config.preprocessed_gaf_file_path,
                                available_nodes_in_string=available_nodes_in_string,
                                available_features_in_data=available_features_in_data,
                                max_level=config.max_level,
                                verbose=True)


    # extract total nodes per level
    nodes_per_level = []
    for i in range(len(pathway_net.layer_indices) - 1):
        nodes_per_level.append(pathway_net.layer_indices[i+1] - pathway_net.layer_indices[i])


    # create input mapping (proteins are input)
    protein_level_idx = len(nodes_per_level) - 1  # last level is proteins
    protein_dim = nodes_per_level[protein_level_idx]

    # input dim depends on the data types specified in the config
    if config.data_type == 'combined':
        number_of_features_per_gene = 2  # CNV + mutation
    else:
        number_of_features_per_gene = 1  # only one data type
    input_dim = protein_dim * number_of_features_per_gene

    # hidden layers are protein layer + specific to general GO terms (excluding most general which will connect to output)
    # we need to reverse (since original order is general to specific)
    hidden_dims = [protein_dim]  # start with protein layer
    for i in reversed(range(0, protein_level_idx)):
        hidden_dims.append(nodes_per_level[i])


    # set output dim
    output_dim = config.output_dim

    # create mapping between pathway indices and network indices
    pathway_to_network_idx = {}
    network_idx = input_dim  # start after input layer

    # map proteins to first hidden layer
    for i in range(pathway_net.layer_indices[protein_level_idx], pathway_net.layer_indices[protein_level_idx+1]):
        pathway_to_network_idx[i] = network_idx
        network_idx += 1

    # map GO term layers from specific to general 
    for level in reversed(range(0, protein_level_idx)):
        for i in range(pathway_net.layer_indices[level], pathway_net.layer_indices[level+1]):
            pathway_to_network_idx[i] = network_idx
            network_idx += 1

    # convert edge indices using the mapping (only for gene→GO and GO→GO connections)
    sources = []
    targets = []

    # add input→gene connections
    # each gene connects to exactly number_of_features_per_gene input nodes
    for protein_idx in range(protein_dim):
        protein_network_idx = input_dim + protein_idx

        feature_indices = [protein_dim * i + protein_idx for i in range(number_of_features_per_gene)]

        for feature_idx in feature_indices:
            sources.append(feature_idx)
            targets.append(protein_network_idx)

    # convert edge indices using the mapping (only for gene→GO and GO→GO connections)  
    for s, t in zip(pathway_net.edge_index[0], pathway_net.edge_index[1]):
        if s in pathway_to_network_idx and t in pathway_to_network_idx:
            sources.append(pathway_to_network_idx[s])
            targets.append(pathway_to_network_idx[t])


    # add connections from last hidden layer to output layer
    last_hidden_layer_start = sum(hidden_dims[:-1]) + input_dim
    last_hidden_layer_size = hidden_dims[-1]
    output_layer_start = sum(hidden_dims) + input_dim

    # connect every node in the last hidden layer to every output node
    for i in range(last_hidden_layer_size):
        source_idx = last_hidden_layer_start + i

        # connect to each output node 
        for j in range(output_dim):
            target_idx = output_layer_start + j

            sources.append(source_idx)
            targets.append(target_idx)


    edge_index = torch.tensor([sources, targets])

    # analyze the edge index
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    analyze_hierarchical_edge_index(edge_index, layer_dims)

    # max index in the edge index
    max_index = max(edge_index[0].max(), edge_index[1].max())
    print(f"Max index in edge index: {max_index.item()}")  

    # total nodes in the network
    total_nodes = pathway_net.graph.number_of_nodes()
    print(f"Total nodes in the network: {total_nodes}")

    return input_dim, hidden_dims, output_dim, edge_index, pathway_net


def create_binn(input_dim : int, 
                hidden_dims : List[int],
                output_dim : int,
                edge_index : torch.Tensor) -> BINN:

    # create neural network
    model = BINN(input_dim=input_dim,
                 hidden_dims=hidden_dims,
                 output_dim=output_dim,
                 norm = 'layer',
                 weight_init='custom')
    
    # set connections
    model.set_connections(edge_index)

    return model

