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
            output = output.squeeze()  # For scalar logit
        grad = torch.autograd.grad(output.sum(), interp)[0]
        attributions += grad
    attributions *= diffs / n_steps
    return attributions


def get_layer_activation(model, binn_input, layer_idx):
    # wrapper to get layer activation (hidden layer_idx, 0 = first hidden)
    _, all_activations = model.forward(binn_input, return_activations=True)
    return all_activations[layer_idx + 1]  # Skip [0] = input, so layer_idx=0 = all_activations[1] = first hidden


def layer_integrated_gradients(model, input_tensor, baseline, layer_idx, target=None, n_steps=50):
    # manual implementation of integrated gradients for a specific hidden layer
    # Get actual layer act
    _, all_activations = model.forward(input_tensor, return_activations=True)
    layer_act = all_activations[layer_idx + 1]  # Adjust for hidden
    _, all_base = model.forward(baseline, return_activations=True)
    layer_baseline_act = all_base[layer_idx + 1]
    diffs = layer_act - layer_baseline_act
    attributions = torch.zeros_like(layer_act)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp_input = baseline + alpha * (input_tensor - baseline)
        interp_layer = get_layer_activation(model, interp_input, layer_idx).detach().requires_grad_(True)
        # Post-layer forward: Rebuild activations up to layer, inject, resume the forward loop
        batch_size = interp_input.size(0)
        activations = torch.zeros(batch_size, model.total_nodes, device=interp_input.device, dtype=interp_input.dtype)
        activations[:, :model.input_dim] = interp_input
        # Fill earlier hidden layers with their interp values
        for j in range(1, layer_idx + 1 + 1):  # i=1 to current hidden i
            start = model.layer_indices[j]
            end = model.layer_indices[j+1]
            activations[:, start:end] = get_layer_activation(model, interp_input, j-1)
        # Inject for current layer
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
        # Resume from next layer (k = layer_idx + 2, where layer_idx is hidden idx starting 0)
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
        # Get output
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


def normalize_scores(scores, model, method='log_subgraph', save_path="normalized_attribution_scores.json"):
    normalized = {}
    # Build graph from sparse_indices (assuming COO [2, num_edges])
    G = nx.DiGraph()
    for src, tgt in model.sparse_indices.t().cpu().numpy():
        G.add_edge(src, tgt)  # Global node indices (source to target, as successors)

    layer_starts = model.layer_indices[:-1]  # Start indices for layers (input + hiddens)
    for layer_key in scores:
        if layer_key == 'end_to_end':
            continue  # Input may not need, or treat as layer 0
        layer_idx = int(layer_key.split('_')[1]) + 1  # hidden_0 = layer 1 (adjust based on your indexing)
        layer_nodes = list(range(layer_starts[layer_idx], layer_starts[layer_idx+1]))
        layer_scores = np.array(scores[layer_key])
        
        if method == 'degree':  # like P-NET
            degrees = np.array([G.degree(n) for n in layer_nodes])
            mean_d, std_d = degrees.mean(), degrees.std()
            mask = degrees > mean_d + 5 * std_d
            layer_scores[mask] /= degrees[mask]
        
        elif method == 'log_subgraph':
            sub_sizes = []
            for n in layer_nodes:
                # Full subgraph: predecessors (ancestors) + successors (descendants) + self
                ancestors = nx.ancestors(G, n)  # Upstream
                descendants = nx.descendants(G, n)  # Downstream
                reachable = ancestors | descendants | {n}
                sub_sizes.append(len(reachable))
            sub_sizes = np.array(sub_sizes)
            layer_scores /= np.log(sub_sizes + 1)  # +1 to avoid log(0)
        
        normalized[layer_key] = layer_scores.tolist()

    with open(save_path, 'w') as f:
        json.dump(normalized, f)

    return normalized


def normalize_scores_with_id(
    scores: Dict[str, torch.Tensor], 
    model: nn.Module, 
    method: str = 'log_subgraph'
) -> Dict[str, np.ndarray]:
    """
    Normalizes attribution scores from the explainer based on the BINN's graph structure.

    Args:
        scores (Dict[str, torch.Tensor]): The raw attribution scores from the explainer.
        model (BINN): The BINN model instance, which contains graph structure info 
                      (edge_index, layer_indices).
        method (str): The normalization method to use ('log_subgraph' or 'degree').

    Returns:
        Dict[str, np.ndarray]: A dictionary with layer names as keys and normalized 
                               scores as NumPy arrays.
    """
    normalized_scores = {}
    
    # Build a graph from the BINN model's edge_index
    G = nx.DiGraph()
    # Assuming the model has an 'edge_index' attribute from its creation
    if hasattr(model, 'edge_index'):
        edges = model.edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
    else:
        raise AttributeError("The provided model does not have an 'edge_index' attribute for graph construction.")

    # Get the start index of each layer in the flattened graph representation
    layer_starts = model.layer_indices

    for layer_key, score_tensor in scores.items():
        # Convert tensor to a NumPy array for processing
        layer_scores_np = score_tensor.detach().cpu().numpy()

        # We only normalize hidden layer scores, as the logic is based on the pathway hierarchy.
        # The 'end_to_end' scores are on the input features and are kept as is.
        if 'hidden' not in layer_key:
            normalized_scores[layer_key] = layer_scores_np
            continue

        try:
            # e.g., 'hidden_0' -> layer 1 (0 is input layer)
            layer_idx = int(layer_key.split('_')[1]) + 1
            # Get the global node indices for the nodes in this specific layer
            layer_nodes_global_indices = list(range(layer_starts[layer_idx], layer_starts[layer_idx + 1]))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse layer index from '{layer_key}'. Skipping normalization for this layer.")
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
            # Add 1 to avoid log(0) for isolated nodes; add epsilon to avoid division by zero.
            layer_scores_np /= (np.log(sub_sizes_np + 1) + 1e-9)
        
        elif method == 'degree':
            degrees = np.array([G.degree(n) for n in layer_nodes_global_indices])
            mean_d, std_d = degrees.mean(), degrees.std()
            # Identify hub nodes
            mask = degrees > mean_d + 5 * std_d
            
            # Normalize scores for hub nodes by their degree
            if np.any(mask):
                # Use a copy to avoid modifying the array while iterating
                temp_scores = layer_scores_np.copy()
                # Add epsilon to avoid division by zero
                temp_scores[:, mask] /= (degrees[mask] + 1e-9)
                layer_scores_np = temp_scores
        else:
            print(f"Warning: Unknown normalization method '{method}'. Scores for '{layer_key}' will not be normalized.")

        normalized_scores[layer_key] = layer_scores_np

    return normalized_scores


def run_sanity_check(gnn_binn_model, binn_model, test_loader, device, n_steps=50, num_runs=5):
    """
    Performs a sanity check on the Integrated Gradients implementation
    by verifying the completeness axiom for a single data sample.

    Axiom: sum(attributions) == model(input) - model(baseline)
    """
    print("--- Running Integrated Gradients Sanity Check ---")

    # Get a single sample from the test loader
    try:
        data_sample = next(iter(test_loader))
    except StopIteration:
        print("Test loader is empty. Cannot run sanity check.")
        return

    # Prepare the single sample for the BINN model
    with torch.no_grad():
        data_sample = data_sample.to(device)
        # We'll just check the first item in the batch
        enriched_features = gnn_binn_model.gnn_feature_extractor(data_sample.x, data_sample.edge_index)
        batch_size = data_sample.batch.max().item() + 1
        reshaped = enriched_features.view(batch_size, gnn_binn_model.num_nodes, gnn_binn_model.gnn_output_features)
        permuted = reshaped.permute(0, 2, 1)
        binn_input_batch = permuted.view(batch_size, -1)

        # Isolate the very first sample and its baseline
        single_input = binn_input_batch[0:1] # Keep batch dim of 1
        single_baseline = torch.zeros_like(single_input)

    # cast the tensors to double precision
    # single_input = single_input.to(dtype=torch.float64)
    # single_baseline = single_baseline.to(dtype=torch.float64)

    print(f"Input tensor shape: {single_input.shape}")

    # Calculate IG attributions for the single sample
    accum = torch.zeros_like(single_input)
    for _ in range(num_runs):  # Average over a few runs for stability
        accum += integrated_gradients(binn_model.forward, single_input, single_baseline, n_steps=n_steps)
    attributions = accum / num_runs

    # Sum the attributions (Left-Hand Side of the axiom)
    sum_of_attributions = attributions.sum().item()

    # Calculate the difference in model output (Right-Hand Side)
    with torch.no_grad():
        output_real = binn_model(single_input).item()
        output_baseline = binn_model(single_baseline).item()
    delta_f = output_real - output_baseline

    # Compare the results
    print(f"\nSum of attributions (LHS):   {sum_of_attributions:.6f}")
    print(f"F(input) - F(baseline) (RHS): {delta_f:.6f}")

    # Calculate the difference and approximation error
    difference = np.abs(sum_of_attributions - delta_f)
    percentage_error = (difference / np.abs(delta_f)) * 100 if delta_f != 0 else 0

    print(f"\nDifference: {difference:.6f}")
    if delta_f != 0:
      print(f"Approximation Error: {percentage_error:.2f}% (due to n_steps={n_steps})")
    
    if percentage_error < 1.0:
        print("\n✅ Sanity check PASSED. The difference is small, as expected.")
    else:
        print("\n⚠️  Sanity check FAILED. The difference is larger than expected.")
    
    print("--- Sanity Check Complete ---")


def explain_model_with_id(gnn_binn_model: GNN_BINN, binn_model: BINN, test_loader: DataLoader, device: torch.device,
                 hidden_dims: List[int], num_runs: int, n_steps: int, save_path: str='attribution_scores.json') -> Dict[str, List[float]]:
    """
    Explains the GNN-BINN model using Integrated Gradients for end-to-end and layer-wise attributions.
    Args:
        gnn_binn_model: The trained GNN-BINN model.
        binn_model: The BINN part of the model.
        test_loader: DataLoader for the test set.
        device: Device to run computations on.
        hidden_dims: List of hidden layer dimensions in the BINN.
    Returns:
        A dictionary with keys 'end_to_end' and 'hidden_{idx}' for layer-wise attributions.
    """

    # create save path directory if it doesn't exist
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = {}

    # Accumulators for the whole test set (initialize with zeros shaped like one batch's attribs)
    end_to_end_global_accum = None
    layer_global_accums = [None for _ in range(len(hidden_dims))]
    num_samples = 0  # Track total samples for final average

    batch_counter = 0
    for data in test_loader:
        print(f"Processing batch {batch_counter}")
        batch_counter += 1
        with torch.no_grad():
            data = data.to(device)
            enriched_features = gnn_binn_model.gnn_feature_extractor(data.x, data.edge_index)
            batch_size = data.batch.max().item() + 1
            reshaped = enriched_features.view(batch_size, gnn_binn_model.num_nodes, gnn_binn_model.gnn_output_features)
            permuted = reshaped.permute(0, 2, 1)
            binn_input = permuted.view(batch_size, -1)
            zeros_baseline = torch.zeros_like(binn_input)

        # cast the tensors to double precision
        # binn_input = binn_input.to(dtype=torch.float64)
        # zeros_baseline = zeros_baseline.to(dtype=torch.float64)
        
        num_samples += batch_size  # Update total samples
        
        # End-to-end IG
        end_to_end_accum = torch.zeros_like(binn_input)  # Per-batch accumulator for runs
        for _ in range(num_runs):
            end_to_end_accum += integrated_gradients(binn_model.forward, binn_input, zeros_baseline, n_steps=n_steps)
        end_to_end_attribs = end_to_end_accum / num_runs  # Average over runs (per batch)
        
        if end_to_end_global_accum is None:
            end_to_end_global_accum = torch.zeros((0,) + end_to_end_attribs.shape[1:], device=device)
        end_to_end_global_accum = torch.cat((end_to_end_global_accum, end_to_end_attribs), dim=0)  # Stack over all samples
        
        # Layer-wise for each hidden layer
        for layer_idx in range(len(hidden_dims)):
            print(f"Processing layer {layer_idx}")
            layer_accum = None  # Per-batch run accumulator
            for _ in range(num_runs):
                print(f"  Run {_+1}/{num_runs} for layer {layer_idx}")
                layer_attrib = layer_integrated_gradients(binn_model, binn_input, zeros_baseline, layer_idx, n_steps=n_steps)
                if layer_accum is None:
                    layer_accum = torch.zeros_like(layer_attrib)
                layer_accum += layer_attrib
            layer_attribs = layer_accum / num_runs  # Average over runs (per batch)
            
            if layer_global_accums[layer_idx] is None:
                layer_global_accums[layer_idx] = torch.zeros((0,) + layer_attribs.shape[1:], device=device)
            layer_global_accums[layer_idx] = torch.cat((layer_global_accums[layer_idx], layer_attribs), dim=0)  # Stack over all samples

            print(f"Processed layer {layer_idx} for batch {batch_counter}")

        print(f"Completed batch {batch_counter}")



    # After all batches: Compute final means over all samples
    scores['end_to_end'] = end_to_end_global_accum.cpu() # Shape: (n_samples, n_features)
    for layer_idx in range(len(hidden_dims)):
        scores[f'hidden_{layer_idx}'] = layer_global_accums[layer_idx].cpu() # Shape: (n_samples, n_layer_nodes)

    return scores


def explain_model(gnn_binn_model: GNN_BINN, binn_model: BINN, test_loader: DataLoader, device: torch.device,
                 hidden_dims: List[int], num_runs: int, n_steps: int, save_path: str='attribution_scores.json') -> Dict[str, List[float]]:
    """
    Explains the GNN-BINN model using Integrated Gradients for end-to-end and layer-wise attributions.
    Args:
        gnn_binn_model: The trained GNN-BINN model.
        binn_model: The BINN part of the model.
        test_loader: DataLoader for the test set.
        device: Device to run computations on.
        hidden_dims: List of hidden layer dimensions in the BINN.
    Returns:
        A dictionary with keys 'end_to_end' and 'hidden_{idx}' for layer-wise attributions.
    """

    # create save path directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = {}

    # Accumulators for the whole test set (initialize with zeros shaped like one batch's attribs)
    end_to_end_global_accum = None
    layer_global_accums = [None for _ in range(len(hidden_dims))]
    num_samples = 0  # Track total samples for final average

    batch_counter = 0
    for data in test_loader:
        print(f"Processing batch {batch_counter}")
        batch_counter += 1
        with torch.no_grad():
            data = data.to(device)
            enriched_features = gnn_binn_model.gnn_feature_extractor(data.x, data.edge_index)
            batch_size = data.batch.max().item() + 1
            reshaped = enriched_features.view(batch_size, gnn_binn_model.num_nodes, gnn_binn_model.gnn_output_features)
            permuted = reshaped.permute(0, 2, 1)
            binn_input = permuted.view(batch_size, -1)
            zeros_baseline = torch.zeros_like(binn_input)

        # cast the tensors to double precision
        # binn_input = binn_input.to(dtype=torch.float64)
        # zeros_baseline = zeros_baseline.to(dtype=torch.float64)
        
        num_samples += batch_size  # Update total samples
        
        # End-to-end IG
        end_to_end_accum = torch.zeros_like(binn_input)  # Per-batch accumulator for runs
        for _ in range(num_runs):
            end_to_end_accum += integrated_gradients(binn_model.forward, binn_input, zeros_baseline, n_steps=n_steps)
        end_to_end_attribs = end_to_end_accum / num_runs  # Average over runs (per batch)
        
        if end_to_end_global_accum is None:
            end_to_end_global_accum = torch.zeros((0,) + end_to_end_attribs.shape[1:], device=device)
        end_to_end_global_accum = torch.cat((end_to_end_global_accum, end_to_end_attribs), dim=0)  # Stack over all samples
        
        # Layer-wise for each hidden layer
        for layer_idx in range(len(hidden_dims)):
            print(f"Processing layer {layer_idx}")
            layer_accum = None  # Per-batch run accumulator
            for _ in range(num_runs):
                print(f"  Run {_+1}/{num_runs} for layer {layer_idx}")
                layer_attrib = layer_integrated_gradients(binn_model, binn_input, zeros_baseline, layer_idx, n_steps=n_steps)
                if layer_accum is None:
                    layer_accum = torch.zeros_like(layer_attrib)
                layer_accum += layer_attrib
            layer_attribs = layer_accum / num_runs  # Average over runs (per batch)
            
            if layer_global_accums[layer_idx] is None:
                layer_global_accums[layer_idx] = torch.zeros((0,) + layer_attribs.shape[1:], device=device)
            layer_global_accums[layer_idx] = torch.cat((layer_global_accums[layer_idx], layer_attribs), dim=0)  # Stack over all samples

            print(f"Processed layer {layer_idx} for batch {batch_counter}")

        print(f"Completed batch {batch_counter}")



    # After all batches: Compute final means over all samples
    scores['end_to_end'] = end_to_end_global_accum.mean(dim=0).cpu().tolist()  # Mean over all samples

    for layer_idx in range(len(hidden_dims)):
        scores[f'hidden_{layer_idx}'] = layer_global_accums[layer_idx].mean(dim=0).cpu().tolist()


    with open(save_path, 'w') as f:
        json.dump(scores, f)

    return scores


def map_scores_to_nodes(scores: Dict[str, List[float]], pathway_net: PathwayNetwork, 
                        save_path: str = 'mapped_attribution_scores.json') -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Maps attribution scores to node IDs and human-readable names per layer.
    
    Args:
        scores: Dict with keys 'end_to_end' and 'hidden_{idx}', values as list of scores per node.
        pathway_net: Instance of PathwayNetwork with layer_indices, index_to_node, id_to_name.
    
    Returns:
        Dict[layer_key, list of (node_id, name, score)] sorted by abs(score) descending.
    """
    # create save path directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    mapped = {}
    # Calculate the number of layers and the protein layer index
    num_layers = len(pathway_net.layer_indices) - 1
    protein_layer_idx = num_layers - 1
    
    for key, score_list in scores.items():
        if key == 'end_to_end':
            # End-to-end scores are for the input features (enriched by GNN)
            # Map to the protein layer, aggregating over features per protein if multiple
            pathway_layer = protein_layer_idx
            layer_start = pathway_net.layer_indices[pathway_layer]
            layer_end = pathway_net.layer_indices[pathway_layer + 1]
            layer_size = layer_end - layer_start
            score_len = len(score_list)
            if score_len % layer_size != 0:
                raise ValueError(f"Score list length {score_len} is not divisible by layer size {layer_size} for {key}")
            num_feats = score_len // layer_size
            # Reshape and sum over features (absolute sum for importance)
            aggregated_scores = [0.0] * layer_size
            for i in range(layer_size):
                node_scores = [score_list[f * layer_size + i] for f in range(num_feats)]
                aggregated_scores[i] = sum(abs(s) for s in node_scores)  # Or sum(node_scores) if signed importance
        else:
            # Hidden layers: Reverse the mapping since BINN order is specific to general
            hidden_idx = int(key.split('_')[1])
            pathway_layer = protein_layer_idx - hidden_idx
            if pathway_layer < 0:
                raise ValueError(f"Invalid pathway layer {pathway_layer} for {key}")
            layer_start = pathway_net.layer_indices[pathway_layer]
            layer_end = pathway_net.layer_indices[pathway_layer + 1]
            layer_size = layer_end - layer_start
            if len(score_list) != layer_size:
                raise ValueError(f"Score list length {len(score_list)} doesn't match layer size {layer_size} for {key}")
            aggregated_scores = score_list  # Direct mapping for hidden layers
        
        # Now map the (aggregated) scores to nodes
        layer_nodes = []
        for i in range(layer_start, layer_end):
            node_id = pathway_net.index_to_node.get(i, f"Unknown_{i}")
            name = pathway_net.id_to_name.get(node_id, node_id)  # Fallback to ID if no name
            score = aggregated_scores[i - layer_start]
            layer_nodes.append((node_id, name, score))
        
        # Sort by abs(score) descending
        layer_nodes.sort(key=lambda x: abs(x[2]), reverse=True)
        mapped[key] = layer_nodes

    with open(save_path, 'w') as f:
        json.dump(mapped, f, indent=4)

    return mapped


def print_top_k_nodes(mapped_scores: Dict[str, List[Tuple[str, str, float]]], k: int = 10):
    """
    Prints the top k most important nodes (by abs(score)) for each layer.
    
    Args:
        mapped_scores: Output from map_scores_to_nodes.
        k: Number of top nodes to print per layer (default 10).
    """
    for layer_key, nodes in mapped_scores.items():
        print(f"Top {k} most important nodes in {layer_key} (by abs(score)):")
        for node_id, name, score in nodes[:k]:
            print(f"  {node_id} ({name}): {score:.6f}")
        print("\n")  


    

