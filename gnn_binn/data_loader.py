import sys
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import StratifiedKFold


from config.gnn_binn_config import GNNBinnConfig
from config.binn_config import BinnConfig
from binn.src.model.pathway_network import PathwayNetwork
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class GNNProstateCancerDataset(Dataset):
    # a pytorch geometric dataset for creating graph-based representations of patient data.
    #
    # each sample in the dataset is a graph, where the node features are derived
    # from a patient's cnv/mutation data and the graph structure is shared.
    def __init__(self, X: np.ndarray, y: np.ndarray, edge_index: torch.Tensor, 
                 num_nodes: int, num_features_per_node: int):
        # initializes the dataset.
        super().__init__()
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_features_per_node = num_features_per_node


    def len(self) -> int:
        # returns the number of samples in the dataset.
        return len(self.y)


    def get(self, idx: int) -> Data:
        # gets a single graph data object for a given index.
        #
        # this method processes one patient's data, reshapes the feature vector into a 
        # node feature matrix, and combines it with the graph structure.
        
        # get the flattened feature vector for a single patient
        patient_features_flat = torch.tensor(self.X[idx], dtype=torch.float)
        
        # reshape the vector into a node feature matrix [num_nodes, num_features_per_node]
        node_features = patient_features_flat.reshape(self.num_nodes, self.num_features_per_node)
        
        # get the corresponding label
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0) # shape [1]
        
        # create a pytorch geometric data object
        graph_data = Data(x=node_features, edge_index=self.edge_index, y=label)
        
        return graph_data
    

def create_cv_dataloaders(X, y, edge_index, num_nodes, num_features_per_node, config, seed=None):
    # creates k-fold dataloaders for pure cv: yields train_loader, val_loader, test_loader, pos_w per fold.
    if seed is None:
        seed = config.random_seed
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- preparing fold {fold + 1}/{config.num_folds} ---")
        
        # split data for the current fold
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        # sub-split train_val into train and val 
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1 / (1 - 1/config.num_folds), 
            random_state=config.random_seed, stratify=y_train_val
        )
        
        # create datasets
        train_dataset = GNNProstateCancerDataset(X_train, y_train, edge_index, num_nodes, num_features_per_node)
        val_dataset = GNNProstateCancerDataset(X_val, y_val, edge_index, num_nodes, num_features_per_node)
        test_dataset = GNNProstateCancerDataset(X_test, y_test, edge_index, num_nodes, num_features_per_node)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # pos weight on train
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"class weights for fold {fold + 1}: {class_weights}, pos weight: {pos_weight.item()}")
        
        # print sizes
        print(f"train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}")
        print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")
        
        yield train_loader, val_loader, test_loader, pos_weight


def create_cv_dataloaders_with_id(X, y, edge_index, num_nodes, num_features_per_node, config, seed=None):
    # creates k-fold dataloaders for pure cv.
    # yields train_loader, val_loader, test_loader, pos_w, and test_indices per fold.
    # returns the original test indices to track samples.
    if seed is None:
        seed = config.random_seed

    print(f"seeding stratifiedkfold with seed: {seed}")
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=seed)
    
    # we need the original indices to track samples
    original_indices = np.arange(len(y))

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- preparing fold {fold + 1}/{config.num_folds} ---")
        
        # split data for the current fold
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        # sub-split train_val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1 / (1 - 1/config.num_folds),
            random_state=config.random_seed, stratify=y_train_val
        )
        
        # create datasets
        train_dataset = GNNProstateCancerDataset(X_train, y_train, edge_index, num_nodes, num_features_per_node)
        val_dataset = GNNProstateCancerDataset(X_val, y_val, edge_index, num_nodes, num_features_per_node)
        test_dataset = GNNProstateCancerDataset(X_test, y_test, edge_index, num_nodes, num_features_per_node)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # pos weight on train
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"class weights for fold {fold + 1}: {class_weights}, pos weight: {pos_weight.item()}")
        
        # print sizes
        print(f"train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}")
        
        original_test_indices = original_indices[test_idx]
        yield train_loader, val_loader, test_loader, pos_weight, original_test_indices


def align_data_robust(X_data: np.ndarray, 
                        data_feature_names: List[str], 
                        network_protein_order: List[str]) -> Tuple[np.ndarray, List[str]]:
    # aligns the columns of a data matrix to a specified network order.
    #
    # this version is more robust as it uses an explicit list of feature names
    # instead of relying on the index of a dataframe.
    
    # create a mapping from the protein id to its column index in the input data
    data_protein_to_col_idx = {
        str(name).strip().upper(): i 
        for i, name in enumerate(data_feature_names)
    }
    print(f"available proteins in data source: {len(data_protein_to_col_idx)}")

    aligned_feature_indices = []
    aligned_feature_names = []
    missing_proteins = []

    # iterate through proteins in the network's desired order
    for network_protein_id in network_protein_order:
        normalized_network_id = str(network_protein_id).strip().upper()
        
        if normalized_network_id in data_protein_to_col_idx:
            # if protein exists in our data, grab its original column index
            data_col_idx = data_protein_to_col_idx[normalized_network_id]
            aligned_feature_indices.append(data_col_idx)
            aligned_feature_names.append(network_protein_id)
        else:
            missing_proteins.append(network_protein_id)
    
    print(f"successfully aligned {len(aligned_feature_indices)} proteins.")
    if len(missing_proteins) > 0:
        print(f"missing from data: {len(missing_proteins)} proteins.")

    # reorder the columns of the original data matrix based on the network order
    X_aligned = X_data[:, aligned_feature_indices]
    print(f"shape of aligned x: {X_aligned.shape}")

    return X_aligned, aligned_feature_names


def verify_alignment(pathway_net, aligned_protein_names):
    # verify that data columns match network input node order
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    
    print("\nverifying alignment:")
    for i, expected_protein in enumerate(aligned_protein_names[:10]):  # check first 10
        pathway_idx = protein_start_idx + i
        network_node = pathway_net.index_to_node[pathway_idx]
        network_protein = network_node.replace("PROTEIN:", "")
        
        matches = (expected_protein.upper().strip() == network_protein.upper().strip())
        status = "✓" if matches else "✗"
        print(f"  input[{i}]: data={expected_protein} | network={network_protein} {status}")


def load_full_aligned_data(binn_config, gnn_binn_config, pathway_net, loaded_data, features):
    # adapted from create_aligned_dataloaders: align full data, no splits. returns x_full, y_full, edge_index, pos_weight, network_ordered_protein_ids.
    
    print("loading graph structure for gnn...")
    # load the edges file you pre-processed
    edges = np.load(gnn_binn_config.filtered_ordered_edges_path)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"loaded {edge_index.shape[1]} edges.")

    # determine the number of features per node based on the data type
    if binn_config.data_type == 'combined':
        num_features_per_node = 2
    else:
        num_features_per_node = 1
    print(f"each node will have {num_features_per_node} feature(s).")

    # get the ordered list of protein ids from the network's input layer
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    network_ordered_protein_ids = [
        pathway_net.index_to_node[i].replace("PROTEIN:", "")
        for i in range(protein_start_idx, protein_end_idx)
    ]
    num_nodes = len(network_ordered_protein_ids)
    print(f"network input layer expects {num_nodes} proteins (nodes).")

    print("aligning full raw data...")
    data_subsets, y_full = loaded_data['data_subsets'], loaded_data['y']  
    if binn_config.data_type == 'combined':
        X_cnv, _, cnv_genes = data_subsets['cnv']
        X_mut, _, mut_genes = data_subsets['mut']
        X_cnv_aligned, cnv_aligned_names = align_data_robust(X_cnv, cnv_genes, network_ordered_protein_ids)
        X_mut_aligned, _ = align_data_robust(X_mut, mut_genes, network_ordered_protein_ids)
        X_full = np.concatenate([X_cnv_aligned, X_mut_aligned], axis=1)
        verify_alignment(pathway_net, cnv_aligned_names)
    else:
        X_filtered, _, input_genes = data_subsets[binn_config.data_type]
        X_full, aligned_names = align_data_robust(X_filtered, input_genes, network_ordered_protein_ids)
        verify_alignment(pathway_net, aligned_names)
    
    # load edges (from your code)
    edges = np.load(gnn_binn_config.filtered_ordered_edges_path)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # pos weight on full
    unique_classes = np.unique(y_full)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_full.flatten())
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
    
    if binn_config.save_data_splits:  
        np.savez(binn_config.data_split_save_path.replace('.npz', '_full.npz'), X_full=X_full, y_full=y_full, pos_weight=pos_weight.numpy())
    
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    network_ordered_protein_ids = [pathway_net.index_to_node[i].replace("PROTEIN:", "") for i in range(protein_start_idx, protein_end_idx)]
    
    return X_full, y_full, edge_index, pos_weight, network_ordered_protein_ids