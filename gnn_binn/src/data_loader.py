import sys
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.gnn_binn_config import GNNBinnConfig
from config.binn_config import BinnConfig
from binn.src.model.pathway_network import PathwayNetwork
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class GNNProstateCancerDataset(Dataset):
    """
    A PyTorch Geometric Dataset for creating graph-based representations of patient data.

    Each sample in the dataset is a graph, where the node features are derived
    from a patient's CNV/mutation data and the graph structure is shared.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, edge_index: torch.Tensor, 
                 num_nodes: int, num_features_per_node: int):
        """
        Initializes the dataset.

        Args:
            X (np.ndarray): The feature matrix where each row is a patient and columns 
                            are flattened gene features. Shape: (num_samples, num_nodes * num_features_per_node).
            y (np.ndarray): The labels for each patient. Shape: (num_samples,).
            edge_index (torch.Tensor): The shared graph connectivity in COO format. Shape: [2, num_edges].
            num_nodes (int): The total number of nodes (genes) in the graph.
            num_features_per_node (int): The number of features for each node (e.g., 1 or 2).
        """
        super().__init__()
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_features_per_node = num_features_per_node

    def len(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.y)

    def get(self, idx: int) -> Data:
        """
        Gets a single graph data object for a given index.

        This method processes one patient's data, reshapes the feature vector into a 
        node feature matrix, and combines it with the graph structure.
        """
        # Get the flattened feature vector for a single patient
        patient_features_flat = torch.tensor(self.X[idx], dtype=torch.float)
        
        # Reshape the vector into a node feature matrix [num_nodes, num_features_per_node]
        node_features = patient_features_flat.reshape(self.num_nodes, self.num_features_per_node)
        
        # Get the corresponding label
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)
        
        # Create a PyTorch Geometric Data object
        graph_data = Data(x=node_features, edge_index=self.edge_index, y=label)
        
        return graph_data
    

def create_cv_dataloaders(X, y, edge_index, num_nodes, num_features_per_node, config, seed=None):
    """Creates k-fold dataloaders for pure CV: Yields train_loader, val_loader, test_loader, pos_w per fold."""
    if seed is None:
        seed = config.random_seed
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Preparing Fold {fold + 1}/{config.num_folds} ---")
        
        # Split data for the current fold
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1 / (1 - 1/config.num_folds),  
            random_state=config.random_seed, stratify=y_train_val
        )
        
        # Create datasets
        train_dataset = GNNProstateCancerDataset(X_train, y_train, edge_index, num_nodes, num_features_per_node)
        val_dataset = GNNProstateCancerDataset(X_val, y_val, edge_index, num_nodes, num_features_per_node)
        test_dataset = GNNProstateCancerDataset(X_test, y_test, edge_index, num_nodes, num_features_per_node)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Pos weight on train
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"Class weights for fold {fold + 1}: {class_weights}, Pos weight: {pos_weight.item()}")
        
        # Print sizes
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        yield train_loader, val_loader, test_loader, pos_weight


def create_cv_dataloaders_with_id(X, y, edge_index, num_nodes, num_features_per_node, config, seed=None):
    """
    Creates k-fold dataloaders for pure CV.
    Yields train_loader, val_loader, test_loader, pos_w, AND test_indices per fold.
    """
    if seed is None:
        seed = config.random_seed

    print(f"Seeding StratifiedKFold with seed: {seed}")
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=seed)
    
    # We need the original indices to track samples
    original_indices = np.arange(len(y))

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Preparing Fold {fold + 1}/{config.num_folds} ---")
        
        # Split data for the current fold
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        # Sub-split train_val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1 / (1 - 1/config.num_folds),
            random_state=config.random_seed, stratify=y_train_val
        )
        
        # Create datasets
        train_dataset = GNNProstateCancerDataset(X_train, y_train, edge_index, num_nodes, num_features_per_node)
        val_dataset = GNNProstateCancerDataset(X_val, y_val, edge_index, num_nodes, num_features_per_node)
        test_dataset = GNNProstateCancerDataset(X_test, y_test, edge_index, num_nodes, num_features_per_node)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Pos weight on train
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"Class weights for fold {fold + 1}: {class_weights}, Pos weight: {pos_weight.item()}")
        
        # Print sizes
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        
        original_test_indices = original_indices[test_idx]
        yield train_loader, val_loader, test_loader, pos_weight, original_test_indices


def create_kfold_dataloaders(X, y, edge_index, num_nodes, num_features_per_node, config): 
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.random_seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{config.num_folds} ---")
        
        # Split data for the current fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create PyTorch Geometric datasets
        train_dataset = GNNProstateCancerDataset(X_train, y_train, edge_index, num_nodes, num_features_per_node)
        val_dataset = GNNProstateCancerDataset(X_val, y_val, edge_index, num_nodes, num_features_per_node)
        
        # Create PyTorch Geometric dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # calculate class weights for imbalanced data
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"Class weights for fold {fold + 1}: {class_weights}, Pos weight: {pos_weight.item()}")

        # print dataset sizes
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        yield train_loader, val_loader, pos_weight


def align_data_robust(X_data: np.ndarray, 
                        data_feature_names: List[str], 
                        network_protein_order: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Aligns the columns of a data matrix to a specified network order.

    This version is more robust as it uses an explicit list of feature names
    instead of relying on the index of a DataFrame.

    Args:
        X_data: The input data matrix (samples x features).
        data_feature_names: A list of gene/protein names corresponding to the
                            columns of X_data.
        network_protein_order: The desired order of proteins for the output matrix.

    Returns:
        A tuple containing:
        - X_aligned (np.ndarray): The data matrix with columns filtered and reordered.
        - aligned_feature_names (List[str]): The list of feature names in the new order.
    """
    # Create a mapping from the protein ID to its column index in the input data
    data_protein_to_col_idx = {
        str(name).strip().upper(): i 
        for i, name in enumerate(data_feature_names)
    }
    print(f"Available proteins in data source: {len(data_protein_to_col_idx)}")

    aligned_feature_indices = []
    aligned_feature_names = []
    missing_proteins = []

    # Iterate through proteins in the network's desired order
    for network_protein_id in network_protein_order:
        normalized_network_id = str(network_protein_id).strip().upper()
        
        if normalized_network_id in data_protein_to_col_idx:
            # If protein exists in our data, grab its original column index
            data_col_idx = data_protein_to_col_idx[normalized_network_id]
            aligned_feature_indices.append(data_col_idx)
            aligned_feature_names.append(network_protein_id)
        else:
            missing_proteins.append(network_protein_id)
    
    print(f"Successfully aligned {len(aligned_feature_indices)} proteins.")
    if len(missing_proteins) > 0:
        print(f"Missing from data: {len(missing_proteins)} proteins.")

    # Reorder the columns of the original data matrix based on the network order
    X_aligned = X_data[:, aligned_feature_indices]
    print(f"Shape of aligned X: {X_aligned.shape}")

    return X_aligned, aligned_feature_names


def verify_alignment(pathway_net, aligned_protein_names):
    """Verify that data columns match network input node order"""
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    
    print("\nVerifying alignment:")
    for i, expected_protein in enumerate(aligned_protein_names[:10]):  # Check first 10
        pathway_idx = protein_start_idx + i
        network_node = pathway_net.index_to_node[pathway_idx]
        network_protein = network_node.replace("PROTEIN:", "")
        
        matches = (expected_protein.upper().strip() == network_protein.upper().strip())
        status = "✓" if matches else "✗"
        print(f"  Input[{i}]: Data={expected_protein} | Network={network_protein} {status}")


def create_aligned_dataloaders(
        binn_config: BinnConfig,
        gnn_binn_config: GNNBinnConfig,
        pathway_net: PathwayNetwork,
        loaded_data: Dict[str, Any],
        features: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, List[str]]:
        """
        Aligns data and creates PyTorch Geometric DataLoaders for the GNN.
        """
        print("\nStep 2: Aligning data and creating GNN dataloaders...")

        # Load the edges file you pre-processed
        edges = np.load(gnn_binn_config.filtered_ordered_edges_path)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        print(f"Loaded {edge_index.shape[1]} edges.")

        # Determine the number of features per node based on the data type
        if binn_config.data_type == 'combined':
            num_features_per_node = 2
        else:
            num_features_per_node = 1
        print(f"Each node will have {num_features_per_node} feature(s).")

        protein_level_idx = len(pathway_net.layer_indices) - 2
        protein_start_idx = pathway_net.layer_indices[protein_level_idx]
        protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
        network_ordered_protein_ids = [
            pathway_net.index_to_node[i].replace("PROTEIN:", "")
            for i in range(protein_start_idx, protein_end_idx)
        ]
        num_nodes = len(network_ordered_protein_ids)
        print(f"Network input layer expects {num_nodes} proteins (nodes).")

        if binn_config.load_data_splits:
            print(f"Using pre-saved, aligned data splits, loading from file {binn_config.data_split_save_path}...")
            X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
            X_val, y_val = loaded_data['X_val'], loaded_data['y_val']
            X_test, y_test = loaded_data['X_test'], loaded_data['y_test']
            
            if 'pos_weight' in loaded_data:
                pos_weight = torch.tensor(loaded_data['pos_weight'], dtype=torch.float32)
            else:
                unique_classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
                pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)

        else: # This path is for loading from raw files
            # ... (your data alignment logic remains exactly the same) ...
            # After this block, you will have X_train, y_train, etc. as before.
            print("Aligning raw data and creating new splits...")
            data_subsets, y = loaded_data['data_subsets'], loaded_data['y']

            if binn_config.data_type == 'combined':
                X_cnv, _, cnv_genes = data_subsets['cnv']
                X_mut, _, mut_genes = data_subsets['mut']
                
                X_cnv_aligned, cnv_aligned_names = align_data_robust(X_cnv, cnv_genes, network_ordered_protein_ids)
                X_mut_aligned, _ = align_data_robust(X_mut, mut_genes, network_ordered_protein_ids)
                
                X_aligned = np.concatenate([X_cnv_aligned, X_mut_aligned], axis=1)
                verify_alignment(pathway_net, cnv_aligned_names)
            else:
                X_filtered, _, input_genes = data_subsets[binn_config.data_type]
                X_aligned, aligned_names = align_data_robust(X_filtered, input_genes, network_ordered_protein_ids)
                verify_alignment(pathway_net, aligned_names)
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_aligned, y, test_size=gnn_binn_config.test_size, random_state=gnn_binn_config.random_seed, stratify=y)
            val_size_adj = gnn_binn_config.val_size / (1.0 - gnn_binn_config.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adj, random_state=gnn_binn_config.random_seed, stratify=y_temp)

            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)


            if binn_config.save_data_splits:
                print(f"Saving aligned data splits to {binn_config.data_split_save_path}...") 
                np.savez(binn_config.data_split_save_path, X_train=X_train, y_train=y_train, X_val=X_val,
                        y_val=y_val, X_test=X_test, y_test=y_test, pos_weight=pos_weight.numpy())
                
        train_dataset = GNNProstateCancerDataset(X_train, y_train, edge_index, num_nodes, num_features_per_node)
        val_dataset = GNNProstateCancerDataset(X_val, y_val, edge_index, num_nodes, num_features_per_node)
        test_dataset = GNNProstateCancerDataset(X_test, y_test, edge_index, num_nodes, num_features_per_node)

        train_loader = DataLoader(train_dataset, batch_size=gnn_binn_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=gnn_binn_config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=gnn_binn_config.batch_size, shuffle=False)

        print(f"GNN DataLoaders created: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, pos_weight, network_ordered_protein_ids



def load_full_aligned_data(binn_config, gnn_binn_config, pathway_net, loaded_data, features):
    """Adapted from create_aligned_dataloaders: Align full data, no splits. Returns X_full, y_full, edge_index, pos_weight, network_ordered_protein_ids."""
    
    print("Loading graph structure for GNN...")
    # Load the edges file you pre-processed
    edges = np.load(gnn_binn_config.filtered_ordered_edges_path)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Loaded {edge_index.shape[1]} edges.")

    # Determine the number of features per node based on the data type
    if binn_config.data_type == 'combined':
        num_features_per_node = 2
    else:
        num_features_per_node = 1
    print(f"Each node will have {num_features_per_node} feature(s).")

    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    network_ordered_protein_ids = [
        pathway_net.index_to_node[i].replace("PROTEIN:", "")
        for i in range(protein_start_idx, protein_end_idx)
    ]
    num_nodes = len(network_ordered_protein_ids)
    print(f"Network input layer expects {num_nodes} proteins (nodes).")

    print("Aligning full raw data...")
    data_subsets, y_full = loaded_data['data_subsets'], loaded_data['y']  # Assume y_full is full labels
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
    
    # Load edges (from your code)
    edges = np.load(gnn_binn_config.filtered_ordered_edges_path)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Pos weight on full
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