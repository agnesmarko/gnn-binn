import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import StratifiedKFold


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.binn_config import BinnConfig
from binn.src.model.pathway_network import PathwayNetwork


class ProstateCancerDataset(Dataset):
        """Custom Dataset for Prostate Cancer data."""
        def __init__(self, features, labels):
            # Convert numpy arrays to PyTorch tensors
            self.features = torch.tensor(features, dtype=torch.float32)
            # float targets, shape [batch_size, 1]
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) 

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]


def get_dataset( type : str, features_in: pd.DataFrame, X_in: np.ndarray ):
    feature_indices = features_in["type"] == type
    features_in = features_in[feature_indices].reset_index(drop=True)
    X_in = X_in[:,feature_indices]
 
    # TODO: Currently the uniprot version is not unique, I use the average
 
    features_in['group'] = features_in.agg('-'.join, axis=1)
    groups = features_in['group']
    unique_groups = groups.drop_duplicates(keep='first').reset_index(drop=True)
    group_to_index = {group: idx for idx, group in enumerate(unique_groups)}
    features_in['group_index'] = groups.map(group_to_index)
 
    array_combined = np.zeros((X_in.shape[0], len(unique_groups)))
    for group_idx in range(len(unique_groups)):
        mask = (features_in['group_index'] == group_idx).to_numpy()
        array_combined[:, group_idx] = X_in[:, mask].mean(axis=1)
 
    features_in = features_in.drop_duplicates(subset=["gene", "type"], keep='first').drop(columns=['group', 'group_index']).reset_index(drop=True)
    X_in = array_combined
    print( f"Shape of X: {X_in.shape}" )
 
    input_genes = list(features_in['gene'].unique())
    print( f"Count of input genes: {len(input_genes)}" )
 
    return (X_in, features_in, input_genes)


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


def align_data(X_filtered, features_filtered, network_protein_order):
    # create mapping from protein_id to column index in the filtered data
    data_protein_to_col_idx = {}
    for idx, row in features_filtered.iterrows():
        protein_id = str(row['gene']).strip().upper()
        data_protein_to_col_idx[protein_id] = idx
    
    print(f"Available proteins in data: {len(data_protein_to_col_idx)}")


    # align data
    aligned_feature_indices = []
    aligned_feature_names = []
    missing_proteins = []

    # iterate through proteins in network order
    for network_protein_id in network_protein_order:
        normalized_network_id = str(network_protein_id).strip().upper()
        
        if normalized_network_id in data_protein_to_col_idx:
            # if protein exists in data - add it in network order
            data_col_idx = data_protein_to_col_idx[normalized_network_id]
            aligned_feature_indices.append(data_col_idx)
            aligned_feature_names.append(network_protein_id)
        else:
            missing_proteins.append(network_protein_id)
    
    print(f"Successfully aligned {len(aligned_feature_indices)} proteins")
    print(f"Missing from data: {len(missing_proteins)} proteins")

    # X_aligned columns are in the same order as network input nodes
    X_aligned = X_filtered[:, aligned_feature_indices]
    print(f"Shape of aligned X: {X_aligned.shape}")

    return X_aligned, aligned_feature_names


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


def load_initial_data(config: BinnConfig) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    """
    Loads initial data, either from raw files or pre-saved splits, and returns the gene list.
    """
    print("Step 1: Loading initial data to extract gene list...")
    loaded_data = {}

    # Features are always needed to map data columns to gene names.
    features = pd.read_csv(config.features_path, names=config.features_names, header=0)
    
    if config.load_data_splits:
        print(f"Loading pre-saved data splits from: {config.data_split_save_path}")
        
        splits = np.load(config.data_split_save_path, allow_pickle=True)
        loaded_data = {
            'X_train': splits['X_train'],
            'y_train': splits['y_train'],
            'X_val': splits['X_val'],
            'y_val': splits['y_val'],
            'X_test': splits['X_test'],
            'y_test': splits['y_test'],
        }
        # The 'pos_weight' is often pre-calculated with the split, so we can load it too.
        if 'pos_weight' in splits:
             loaded_data['pos_weight'] = splits['pos_weight']

        print("Pre-saved splits loaded successfully.")
        # When loading splits, all genes are derived directly from the features file,
        # assuming the splits were created using this same set of features.
        all_genes = sorted(list(features['gene'].unique()))

    else:
        print("Loading data from raw files...")
        X = np.load(config.x_path)
        y = np.load(config.y_path)

        data_subsets = {
            'mut': get_dataset("mut_important_plus_hotspots", features, X),
            'cnv': get_dataset("cnv", features, X)
        }
        
        loaded_data = {
            'data_subsets': data_subsets,
            'y': y
        }

        # Extract the list of all unique genes from the loaded feature subsets
        mut_genes = data_subsets['mut'][2]  # The 3rd element is the list of gene names
        cnv_genes = data_subsets['cnv'][2]
        all_genes = sorted(list(set(mut_genes) | set(cnv_genes)))

    print(f"Found {len(all_genes)} unique genes in the dataset.")
    
    return loaded_data, features, all_genes


def create_aligned_dataloaders(
    config: BinnConfig,
    pathway_net: PathwayNetwork,
    loaded_data: Dict[str, Any],
    features: pd.DataFrame
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, List[str]]:
    """
    Aligns data and creates DataLoaders using the robust alignment function.
    """
    print("\nStep 2: Aligning data and creating dataloaders...")

    # Get the ordered list of protein IDs from the network's input layer
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    network_ordered_protein_ids = [
        pathway_net.index_to_node[i].replace("PROTEIN:", "")
        for i in range(protein_start_idx, protein_end_idx)
    ]
    print(f"Network input layer expects {len(network_ordered_protein_ids)} proteins.")

    if config.load_data_splits:
        print("Using pre-saved, aligned data splits. Skipping alignment.")
        
        # Unpack the pre-aligned splits directly from the loaded data
        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_val, y_val = loaded_data['X_val'], loaded_data['y_val']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        # The data is already aligned, so we can proceed directly.
        print("Data splits successfully assigned.")
        
        # Use pre-calculated weight if available, otherwise compute it
        if 'pos_weight' in loaded_data:
            pos_weight = torch.tensor(loaded_data['pos_weight'], dtype=torch.float32)
            print(f"Using pre-loaded positive class weight: {pos_weight.item():.3f}")
        else:
            print("Computing positive class weight from training labels...")
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
            print(f"Computed positive class weight: {pos_weight.item():.3f}")


    else: # This path is for loading from raw files
        print("Aligning raw data and creating new splits...")
        data_subsets, y = loaded_data['data_subsets'], loaded_data['y']

        if config.data_type == 'combined':
            X_cnv, _, cnv_genes = data_subsets['cnv']
            X_mut, _, mut_genes = data_subsets['mut']
            
            X_cnv_aligned, cnv_aligned_names = align_data_robust(X_cnv, cnv_genes, network_ordered_protein_ids)
            X_mut_aligned, _ = align_data_robust(X_mut, mut_genes, network_ordered_protein_ids)
            
            X_aligned = np.concatenate([X_cnv_aligned, X_mut_aligned], axis=1)
            verify_alignment(pathway_net, cnv_aligned_names)
        else:
            X_filtered, _, input_genes = data_subsets[config.data_type]
            X_aligned, aligned_names = align_data_robust(X_filtered, input_genes, network_ordered_protein_ids)
            verify_alignment(pathway_net, aligned_names)
        
        # Create train, validation, and test splits from the newly aligned data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_aligned, y, test_size=config.test_size, random_state=config.random_seed, stratify=y)
        val_size_adj = config.val_size / (1.0 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=config.random_seed, stratify=y_temp)
        
        # Compute class weights
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"Positive class weight computed: {pos_weight.item():.3f}")

        if config.save_data_splits:
            print(f"Saving newly created (and aligned) data splits to {config.data_split_save_path}...")
            # Note: This will now save the ALIGNED data splits.
            np.savez(config.data_split_save_path, X_train=X_train, y_train=y_train, X_val=X_val,
                     y_val=y_val, X_test=X_test, y_test=y_test, pos_weight=pos_weight.numpy())
            print("Save complete.")
    
    # Create final Dataset and DataLoader objects
    train_dataset = ProstateCancerDataset(X_train, y_train)
    val_dataset = ProstateCancerDataset(X_val, y_val)
    test_dataset = ProstateCancerDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"DataLoaders created: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader, pos_weight, network_ordered_protein_ids


def load_full_aligned_data(
    config: BinnConfig,
    pathway_net: PathwayNetwork,
    loaded_data: Dict[str, Any],
    features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, List[str]]:
    """
    Loads and aligns the entire dataset without creating final train/val/test splits.

    This function is designed to prepare the complete dataset for k-fold cross-validation.
    It handles both loading from pre-saved splits (by recombining them) and from raw files.

    Returns:
        Tuple containing the full aligned data X, full labels y, the positive class
        weight calculated on the full dataset, and the ordered list of protein IDs
        from the network's input layer.
    """
    print("\nStep 2: Aligning full dataset for Cross-Validation...")

    # Get the ordered list of protein IDs from the network's input layer
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    network_ordered_protein_ids = [
        pathway_net.index_to_node[i].replace("PROTEIN:", "")
        for i in range(protein_start_idx, protein_end_idx)
    ]
    print(f"Network input layer expects {len(network_ordered_protein_ids)} features.")

    if config.load_data_splits:
        print("Reconstructing full dataset from pre-saved splits...")
        
        # Unpack the pre-aligned splits
        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_val, y_val = loaded_data['X_val'], loaded_data['y_val']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        # Concatenate all parts to get the full dataset
        X_full = np.concatenate([X_train, X_val, X_test], axis=0)
        y_full = np.concatenate([y_train, y_val, y_test], axis=0)
        
        print(f"Reconstructed full dataset with {X_full.shape[0]} samples.")

    else: # This path is for loading from raw files
        print("Aligning raw data to create full dataset...")
        data_subsets, y_full = loaded_data['data_subsets'], loaded_data['y']

        if config.data_type == 'combined':
            X_cnv, _, cnv_genes = data_subsets['cnv']
            X_mut, _, mut_genes = data_subsets['mut']
            
            # Align data robustly according to the network's expected input order
            X_cnv_aligned, cnv_aligned_names = align_data_robust(X_cnv, cnv_genes, network_ordered_protein_ids)
            X_mut_aligned, _ = align_data_robust(X_mut, mut_genes, network_ordered_protein_ids)
            
            X_full = np.concatenate([X_cnv_aligned, X_mut_aligned], axis=1)
            verify_alignment(pathway_net, cnv_aligned_names) # Verify one of the aligned sets
        else:
            X_filtered, _, input_genes = data_subsets[config.data_type]
            X_full, aligned_names = align_data_robust(X_filtered, input_genes, network_ordered_protein_ids)
            verify_alignment(pathway_net, aligned_names)
        
        print(f"Aligned full dataset with {X_full.shape[0]} samples.")

    unique_classes = np.unique(y_full)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_full.flatten())
    pos_weight_full = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
    print(f"Positive class weight for the full dataset: {pos_weight_full.item():.3f}")

    if config.save_data_splits:
        # Save the newly created FULL aligned data
        full_save_path = config.data_split_save_path.replace('.npz', '_full_aligned.npz')
        print(f"Saving newly aligned full dataset to {full_save_path}...")
        np.savez(full_save_path, X_full=X_full, y_full=y_full, pos_weight=pos_weight_full.numpy())
        print("Save complete.")

    return X_full, y_full, pos_weight_full, network_ordered_protein_ids


def create_cv_dataloaders(
    X_full: np.ndarray,
    y_full: np.ndarray,
    config: BinnConfig,
    num_folds: int = 5
):
    """
    Creates a generator for k-fold cross-validation dataloaders.

    For each fold, it yields a training loader, validation loader, test loader,
    and the positive class weight calculated *only* on that fold's training data.

    Args:
        X_full: The complete, aligned feature matrix.
        y_full: The complete label vector.
        config: The BinnConfig object with batch_size, random_seed, etc.
        num_folds: The number of folds to create.

    Yields:
        A tuple of (train_loader, val_loader, test_loader, pos_weight_fold).
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.random_seed)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Preparing Fold {fold + 1}/{num_folds} ---")
        
        # Split data into a training/validation set and a final test set for this fold
        X_train_val, X_test = X_full[train_val_idx], X_full[test_idx]
        y_train_val, y_test = y_full[train_val_idx], y_full[test_idx]
        
        # Adjust validation size relative to the new training/validation set size
        # This ensures the validation set is a fraction of the train_val data, not the original full data
        val_size_adj = config.val_size / (1.0 - (1.0 / num_folds))

        # Sub-split the train_val set into the final training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size_adj, 
            random_state=config.random_seed, 
            stratify=y_train_val
        )

        # Compute positive class weight *only* on the training data for this fold
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight_fold = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"Positive class weight for fold {fold + 1}: {pos_weight_fold.item():.3f}")
        
        # Create Dataset objects for this fold
        train_dataset = ProstateCancerDataset(X_train, y_train)
        val_dataset = ProstateCancerDataset(X_val, y_val)
        test_dataset = ProstateCancerDataset(X_test, y_test)

        # Create DataLoader objects for this fold
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        print(f"Fold {fold + 1} sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"Fold {fold + 1} batches: Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # 7. Yield the loaders and the fold-specific weight
        yield train_loader, val_loader, test_loader, pos_weight_fold


def load_data(config: BinnConfig, pathway_net: PathwayNetwork) -> tuple:

    # get the ORDERED list of protein IDs as they appear in the network input layer
    protein_level_idx = len(pathway_net.layer_indices) - 2  # protein layer
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    
    network_ordered_protein_ids = []
    for pathway_idx in range(protein_start_idx, protein_end_idx):
        protein_node_id = pathway_net.index_to_node[pathway_idx]  # e.g., "PROTEIN:P12345"
        protein_id = protein_node_id.replace("PROTEIN:", "")      # extract "P12345"
        network_ordered_protein_ids.append(protein_id)
    
    print(f"Network input layer expects {len(network_ordered_protein_ids)} proteins in specific order")
    print(f"First 10 network proteins: {network_ordered_protein_ids[:10]}")

    if config.load_data_splits:
        print("Loading pre-saved data splits...")

        splits = np.load(config.data_split_save_path, allow_pickle=True)
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_val = splits['X_val']
        y_val = splits['y_val']
        X_test = splits['X_test']
        y_test = splits['y_test']
        pos_weight = splits['pos_weight']

        print(f"Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"Loaded X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"Loaded X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Loaded positive class weight: {pos_weight}")

    else:
        print("Creating new data splits...")

        # load dataset
        features = pd.read_csv(config.features_path, names=config.features_names, header=0)
        X = np.load(config.x_path)
        y = np.load(config.y_path)

        dataset = {}
        dataset['mut'] = get_dataset( "mut_important_plus_hotspots", features, X )
        dataset['cnv'] = get_dataset( "cnv", features, X )

        # handle different data type configurations
        if config.data_type == 'combined':
            # Multi-modal: combine CNV and mutations
            print("Using combined (multi-modal) data")
            
            X_cnv, features_cnv, _ = dataset['cnv']
            X_mut, features_mut, _ = dataset['mut']

            print(f"Shape of X_cnv: {X_cnv.shape}, Shape of y_cnv: {y.shape}")
            print(f"Shape of X_mut: {X_mut.shape}, Shape of y_mut: {y.shape}")
            print(f" Number of input genes in CNV: {features_cnv.shape[0]}, Number of input genes in Mut: {features_mut.shape[0]}")

            # align both data types to network order
            X_cnv_aligned, cnv_aligned_feature_names = align_data(X_cnv, features_cnv, network_ordered_protein_ids)
            X_mut_aligned, mut_aligned_feature_names = align_data(X_mut, features_mut, network_ordered_protein_ids)

            # Concatenate: [CNV features, Mutation features]
            X_aligned = np.concatenate([X_cnv_aligned, X_mut_aligned], axis=1)
            
            print(f"Combined input shape: {X_aligned.shape}")
            print(f"CNV features: {X_cnv_aligned.shape[1]}, Mutation features: {X_mut_aligned.shape[1]}")

            # verify alignment (only need to check one, since both aligned to same order)
            verify_alignment(pathway_net, cnv_aligned_feature_names)

        else:
            print(f"Using single data type: {config.data_type}")

            # extract the data
            X_filtered, features_filtered, input_genes_filtered = dataset[config.data_type]

            print(f"Shape of X_filtered: {X_filtered.shape}")
            print(f"Shape of y_filtered: {y.shape}")
            print(f"Number of input genes used: {len(input_genes_filtered)}")

            # align data to network order
            X_aligned, aligned_feature_names = align_data(X_filtered, features_filtered, network_ordered_protein_ids)

            # verify alignment
            verify_alignment(pathway_net, aligned_feature_names)

        
        # print class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"Class ratios: {class_counts / len(y)}")

        # create train, val, test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_aligned,
            y,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=y
        )

        val_size_adjusted = config.val_size / (1.0 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=config.random_seed,
            stratify=y_temp
        )

        # compute class weights for loss function
        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_flat)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)

        print(f"Positive class weight: {pos_weight.item():.3f}")

        # save splits for future use
        if config.save_data_splits:
            print("Saving data splits...")
            np.savez(
                config.data_split_save_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                pos_weight=pos_weight
            )
            print(f"Data splits saved to {config.data_split_save_path}")

    # create datasets and dataloaders
    train_dataset = ProstateCancerDataset(X_train, y_train)
    val_dataset = ProstateCancerDataset(X_val, y_val)
    test_dataset = ProstateCancerDataset(X_test, y_test)    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")
    print(f"Number of train batches: {len(train_loader)}, Number of val batches: {len(val_loader)}, Number of test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, pos_weight, network_ordered_protein_ids


