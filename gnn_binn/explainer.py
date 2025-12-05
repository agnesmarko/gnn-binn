import torch
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import torch.nn as nn

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from binn.src.model.pathway_network import PathwayNetwork
from binn.src.model.binn import BINN

from gnn_binn.src.model.gnn_binn import GNN_BINN


def integrated_gradients(model_forward, input_tensor, baseline, target=None, n_steps=50):
    # manual implementation of integrated gradients
    diffs = input_tensor - baseline
    attributions = torch.zeros_like(input_tensor)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp = baseline + alpha * diffs
        interp.requires_grad = True
        output = model_forward(interp)
        if target is not None:
            output = output[:, target] if output.dim() > 1 else output
        else:
            output = output.squeeze()  # for scalar logit
        grad = torch.autograd.grad(output.sum(), interp)[0]
        attributions += grad
    attributions *= diffs / n_steps
    return attributions


def get_layer_activation(model, binn_input, layer_idx):
    # wrapper to get layer activation (hidden layer_idx, 0 = first hidden)
    _, all_activations = model.forward(binn_input, return_activations=True)
    return all_activations[layer_idx + 1]  # skip [0] = input, so layer_idx=0 = all_activations[1] = first hidden


def layer_integrated_gradients(model, input_tensor, baseline, layer_idx, target=None, n_steps=50):
    # manual implementation of integrated gradients for a specific hidden layer
    # get actual layer act
    _, all_activations = model.forward(input_tensor, return_activations=True)
    layer_act = all_activations[layer_idx + 1]  # adjust for hidden
    _, all_base = model.forward(baseline, return_activations=True)
    layer_baseline_act = all_base[layer_idx + 1]
    diffs = layer_act - layer_baseline_act
    attributions = torch.zeros_like(layer_act)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp_input = baseline + alpha * (input_tensor - baseline)
        interp_layer = get_layer_activation(model, interp_input, layer_idx).detach().requires_grad_(True)

        # post-layer forward: rebuild activations up to layer, inject, resume the forward loop
        batch_size = interp_input.size(0)
        activations = torch.zeros(batch_size, model.total_nodes, device=interp_input.device, dtype=interp_input.dtype)
        activations[:, :model.input_dim] = interp_input

        # fill earlier hidden layers with their interp values
        for j in range(1, layer_idx + 1 + 1):  # i=1 to current hidden i
            start = model.layer_indices[j]
            end = model.layer_indices[j+1]
            activations[:, start:end] = get_layer_activation(model, interp_input, j-1)

        # inject for current layer
        layer_start_idx = model.layer_indices[layer_idx + 1]
        layer_end_idx = model.layer_indices[layer_idx + 2]
        activations[:, layer_start_idx:layer_end_idx] = interp_layer
        prev_activations = activations.clone()
        layer_predictions = []
        sparse_indices_dev = model.sparse_indices.to(interp_input.device)
        weights_dev = model.weight.to(interp_input.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (model.total_nodes, model.total_nodes),
            device=interp_input.device
        )
        sparse_weights_t = sparse_weights.t()

        # resume from next layer (k = layer_idx + 2, where layer_idx is hidden idx starting 0)
        for k in range(layer_idx + 2, len(model.layer_dims)):
            k_start = model.layer_indices[k]
            k_end = model.layer_indices[k+1]
            weighted_sum_all = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_weighted = weighted_sum_all[:, k_start:k_end]
            bias_start = k_start - model.input_dim
            bias_end = k_end - model.input_dim
            k_bias = model.bias[bias_start:bias_end].to(interp_input.device)
            if k < len(model.layer_dims) - 1:
                if model.norm == 'layer':
                    normed = model.layer_norms[k-1](current_weighted + k_bias)
                elif model.norm == 'batch':
                    normed = model.batch_norms[k-1](current_weighted + k_bias)
                activated = torch.tanh(normed)
                pred = model.prediction_heads[k-1](activated)
                layer_predictions.append(pred)
                activations[:, k_start:k_end] = activated
            else:
                activated = current_weighted + k_bias
                activations[:, k_start:k_end] = activated
            prev_activations = activations.clone()

        # get output
        if layer_predictions:
            output = torch.stack(layer_predictions).mean(dim=0)
        else:
            output = activations[:, model.layer_indices[-2]:model.layer_indices[-1]]
        if target is not None:
            output = output[:, target] if output.dim() > 1 else output
        else:
            output = output.squeeze()
        grad = torch.autograd.grad(output.sum(), interp_layer)[0]
        attributions += grad
    attributions *= diffs / n_steps
    return attributions


def normalize_scores_with_id(
    scores: Dict[str, torch.Tensor], 
    model: nn.Module, # using generic nn.module, but it's your binn model
    method: str = 'log_subgraph'
) -> Dict[str, np.ndarray]:
    # normalizes attribution scores from the explainer based on the binn's graph structure.
    
    normalized_scores = {}
    
    # build a graph from the binn model's edge_index
    G = nx.DiGraph()
    if hasattr(model, 'edge_index'):
        edges = model.edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
    else:
        raise AttributeError("the provided model does not have an 'edge_index' attribute for graph construction.")

    # get the start index of each layer in the flattened graph representation
    layer_starts = model.layer_indices

    for layer_key, score_tensor in scores.items():
        # convert tensor to a numpy array for processing
        layer_scores_np = score_tensor.detach().cpu().numpy()

        # we only normalize hidden layer scores, as the logic is based on the pathway hierarchy.
        # the 'end_to_end' scores are on the input features and are kept as is.
        if 'hidden' not in layer_key:
            normalized_scores[layer_key] = layer_scores_np
            continue

        try:
            layer_idx = int(layer_key.split('_')[1]) + 1
            # get the global node indices for the nodes in this specific layer
            layer_nodes_global_indices = list(range(layer_starts[layer_idx], layer_starts[layer_idx + 1]))
        except (ValueError, IndexError):
            print(f"warning: could not parse layer index from '{layer_key}'. skipping normalization for this layer.")
            normalized_scores[layer_key] = layer_scores_np
            continue

        if method == 'log_subgraph':
            sub_sizes = []
            for n in layer_nodes_global_indices:
                ancestors = nx.ancestors(G, n)
                descendants = nx.descendants(G, n)
                reachable = ancestors | descendants | {n}
                sub_sizes.append(len(reachable))
            
            sub_sizes_np = np.array(sub_sizes)
            # add 1 to avoid log(0) for isolated nodes; add epsilon to avoid division by zero.
            layer_scores_np /= (np.log(sub_sizes_np + 1) + 1e-9)
        
        elif method == 'degree':
            degrees = np.array([G.degree(n) for n in layer_nodes_global_indices])
            mean_d, std_d = degrees.mean(), degrees.std()
            # identify hub nodes
            mask = degrees > mean_d + 5 * std_d
            
            # normalize scores for hub nodes by their degree
            if np.any(mask):
                # use a copy to avoid modifying the array while iterating
                temp_scores = layer_scores_np.copy()
                # add epsilon to avoid division by zero
                temp_scores[:, mask] /= (degrees[mask] + 1e-9)
                layer_scores_np = temp_scores
        else:
            print(f"warning: unknown normalization method '{method}'. scores for '{layer_key}' will not be normalized.")

        normalized_scores[layer_key] = layer_scores_np

    return normalized_scores


def explain_model_with_id(gnn_binn_model: GNN_BINN, binn_model: BINN, test_loader: DataLoader, device: torch.device,
                 hidden_dims: List[int], num_runs: int, n_steps: int, save_path: str='attribution_scores.json') -> Dict[str, List[float]]:
    # explains the gnn-binn model using integrated gradients for end-to-end and layer-wise attributions.
    

    # create save path directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = {}

    # accumulators for the whole test set (initialize with zeros shaped like one batch's attribs)
    end_to_end_global_accum = None
    layer_global_accums = [None for _ in range(len(hidden_dims))]
    num_samples = 0  # track total samples for final average

    batch_counter = 0
    for data in test_loader:
        print(f"processing batch {batch_counter}")
        batch_counter += 1
        with torch.no_grad():
            data = data.to(device)
            enriched_features = gnn_binn_model.gnn_feature_extractor(data.x, data.edge_index)
            batch_size = data.batch.max().item() + 1
            reshaped = enriched_features.view(batch_size, gnn_binn_model.num_nodes, gnn_binn_model.gnn_output_features)
            permuted = reshaped.permute(0, 2, 1)
            binn_input = permuted.view(batch_size, -1)
            zeros_baseline = torch.zeros_like(binn_input)

        
        num_samples += batch_size  # update total samples
        
        # end-to-end ig
        end_to_end_accum = torch.zeros_like(binn_input)  # per-batch accumulator for runs
        for _ in range(num_runs):
            end_to_end_accum += integrated_gradients(binn_model.forward, binn_input, zeros_baseline, n_steps=n_steps)
        end_to_end_attribs = end_to_end_accum / num_runs  # average over runs (per batch)
        
        if end_to_end_global_accum is None:
            end_to_end_global_accum = torch.zeros((0,) + end_to_end_attribs.shape[1:], device=device)
        end_to_end_global_accum = torch.cat((end_to_end_global_accum, end_to_end_attribs), dim=0)  # stack over all samples
        
        # layer-wise for each hidden layer
        for layer_idx in range(len(hidden_dims)):
            print(f"processing layer {layer_idx}")
            layer_accum = None  # per-batch run accumulator
            for _ in range(num_runs):
                print(f"  run {_+1}/{num_runs} for layer {layer_idx}")
                layer_attrib = layer_integrated_gradients(binn_model, binn_input, zeros_baseline, layer_idx, n_steps=n_steps)
                if layer_accum is None:
                    layer_accum = torch.zeros_like(layer_attrib)
                layer_accum += layer_attrib
            layer_attribs = layer_accum / num_runs  # average over runs (per batch)
            
            if layer_global_accums[layer_idx] is None:
                layer_global_accums[layer_idx] = torch.zeros((0,) + layer_attribs.shape[1:], device=device)
            layer_global_accums[layer_idx] = torch.cat((layer_global_accums[layer_idx], layer_attribs), dim=0)  # stack over all samples

            print(f"processed layer {layer_idx} for batch {batch_counter}")

        print(f"completed batch {batch_counter}")



    # after all batches: compute final means over all samples
    scores['end_to_end'] = end_to_end_global_accum.cpu() # shape: (n_samples, n_features)
    for layer_idx in range(len(hidden_dims)):
        scores[f'hidden_{layer_idx}'] = layer_global_accums[layer_idx].cpu() # shape: (n_samples, n_layer_nodes)

    return scores

                     
    # prints the top k most important nodes (by abs(score)) for each layer.
    
    for layer_key, nodes in mapped_scores.items():
        print(f"top {k} most important nodes in {layer_key} (by abs(score)):")
        for node_id, name, score in nodes[:k]:
            print(f"  {node_id} ({name}): {score:.6f}")
        print("\n")  


    # saves attribution data in json format for sankey diagrams.
    # 
    # args:
    #     mapped_scores: output from map_scores_to_nodes.
    #     pathway_net: for graph edges and mappings.
    #     file_path: output json path.
    # 
    # format: {"nodes": [{"id": str, "name": str, "layer": int, "score": float}, ...],
    #          "links": [{"source": str, "target": str, "value": float}, ...]}
    sankey_data = {"nodes": [], "links": []}
    
    # track all node ids for link filtering
    all_node_ids = set()
    
    # nodes: assign layers starting from 0 (end_to_end/input), increasing for hiddens
    layer_num = 0
    for layer_key, nodes in mapped_scores.items():
        for node_id, name, score in nodes:
            sankey_data["nodes"].append({
                "id": node_id,
                "name": name,
                "layer": layer_num,
                "score": abs(score)  # for color/size in sankey
            })
            all_node_ids.add(node_id)
        layer_num += 1
    
    # links: from graph edges, only if both nodes in mapped_scores
    for source, target in pathway_net.graph.edges():
        if source in all_node_ids and target in all_node_ids:
            # value=1.0; if graph has 'weight' attribute, use data=true in edges()
            sankey_data["links"].append({
                "source": source,
                "target": target,
                "value": 1.0
            })
    
    with open(save_path, 'w') as f:
        json.dump(sankey_data, f, indent=4)