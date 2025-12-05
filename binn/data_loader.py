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


from config.binn_config import BinnConfig
from binn.src.model.pathway_network import PathwayNetwork


class ProstateCancerDataset(Dataset):
    # custom dataset for prostate cancer data.
    def __init__(self, features, labels):
        # convert numpy arrays to pytorch tensors
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
 
    # todo: currently the uniprot version is not unique, i use the average
 
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
    print( f"shape of x: {X_in.shape}" )
 
    input_genes = list(features_in['gene'].unique())
    print( f"count of input genes: {len(input_genes)}" )
 
    return (X_in, features_in, input_genes)


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
        status = "ok" if matches else "not ok"
        print(f"  input[{i}]: data={expected_protein} | network={network_protein} {status}")


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


def load_initial_data(config: BinnConfig) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    # loads initial data, either from raw files or pre-saved splits, and returns the gene list.
    
    print("step 1: loading initial data to extract gene list...")
    loaded_data = {}

    # features are always needed to map data columns to gene names.
    features = pd.read_csv(config.features_path, names=config.features_names, header=0)
    
    if config.load_data_splits:
        print(f"loading pre-saved data splits from: {config.data_split_save_path}")
        
        splits = np.load(config.data_split_save_path, allow_pickle=True)
        loaded_data = {
            'X_train': splits['X_train'],
            'y_train': splits['y_train'],
            'X_val': splits['X_val'],
            'y_val': splits['y_val'],
            'X_test': splits['X_test'],
            'y_test': splits['y_test'],
        }
        # the 'pos_weight' is often pre-calculated with the split, so we can load it too.
        if 'pos_weight' in splits:
             loaded_data['pos_weight'] = splits['pos_weight']

        print("pre-saved splits loaded successfully.")
        # when loading splits, all genes are derived directly from the features file,
        # assuming the splits were created using this same set of features.
        all_genes = sorted(list(features['gene'].unique()))

    else:
        print("loading data from raw files...")
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

        # extract the list of all unique genes from the loaded feature subsets
        mut_genes = data_subsets['mut'][2]  # the 3rd element is the list of gene names
        cnv_genes = data_subsets['cnv'][2]
        all_genes = sorted(list(set(mut_genes) | set(cnv_genes)))

    print(f"found {len(all_genes)} unique genes in the dataset.")
    
    return loaded_data, features, all_genes


def load_full_aligned_data(
    config: BinnConfig,
    pathway_net: PathwayNetwork,
    loaded_data: Dict[str, Any],
    features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, List[str]]:
    # loads and aligns the entire dataset without creating final train/val/test splits.
    #
    # this function is designed to prepare the complete dataset for k-fold cross-validation.
    # it handles both loading from pre-saved splits (by recombining them) and from raw files.
    print("\nstep 2: aligning full dataset for cross-validation...")

    # get the ordered list of protein ids from the network's input layer
    protein_level_idx = len(pathway_net.layer_indices) - 2
    protein_start_idx = pathway_net.layer_indices[protein_level_idx]
    protein_end_idx = pathway_net.layer_indices[protein_level_idx + 1]
    network_ordered_protein_ids = [
        pathway_net.index_to_node[i].replace("PROTEIN:", "")
        for i in range(protein_start_idx, protein_end_idx)
    ]
    print(f"network input layer expects {len(network_ordered_protein_ids)} features.")

    if config.load_data_splits:
        print("reconstructing full dataset from pre-saved splits...")
        
        # unpack the pre-aligned splits
        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_val, y_val = loaded_data['X_val'], loaded_data['y_val']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        # concatenate all parts to get the full dataset
        X_full = np.concatenate([X_train, X_val, X_test], axis=0)
        y_full = np.concatenate([y_train, y_val, y_test], axis=0)
        
        print(f"reconstructed full dataset with {X_full.shape[0]} samples.")

    else: # this path is for loading from raw files
        print("aligning raw data to create full dataset...")
        data_subsets, y_full = loaded_data['data_subsets'], loaded_data['y']

        if config.data_type == 'combined':
            X_cnv, _, cnv_genes = data_subsets['cnv']
            X_mut, _, mut_genes = data_subsets['mut']
            
            # align data robustly according to the network's expected input order
            X_cnv_aligned, cnv_aligned_names = align_data_robust(X_cnv, cnv_genes, network_ordered_protein_ids)
            X_mut_aligned, _ = align_data_robust(X_mut, mut_genes, network_ordered_protein_ids)
            
            X_full = np.concatenate([X_cnv_aligned, X_mut_aligned], axis=1)
            verify_alignment(pathway_net, cnv_aligned_names) # verify one of the aligned sets
        else:
            X_filtered, _, input_genes = data_subsets[config.data_type]
            X_full, aligned_names = align_data_robust(X_filtered, input_genes, network_ordered_protein_ids)
            verify_alignment(pathway_net, aligned_names)
        
        print(f"aligned full dataset with {X_full.shape[0]} samples.")

    # compute class weights on the entire dataset
    # note: for cv, it's better to compute this on the training set of each fold later.
    # however, we can compute a "global" weight here if needed.
    unique_classes = np.unique(y_full)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_full.flatten())
    pos_weight_full = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
    print(f"positive class weight for the full dataset: {pos_weight_full.item():.3f}")

    if config.save_data_splits:
        # save the newly created full aligned data
        full_save_path = config.data_split_save_path.replace('.npz', '_full_aligned.npz')
        print(f"saving newly aligned full dataset to {full_save_path}...")
        np.savez(full_save_path, X_full=X_full, y_full=y_full, pos_weight=pos_weight_full.numpy())
        print("save complete.")

    return X_full, y_full, pos_weight_full, network_ordered_protein_ids


def create_cv_dataloaders(
    X_full: np.ndarray,
    y_full: np.ndarray,
    config: BinnConfig,
    num_folds: int = 5,
    seed=None
):
    # creates a generator for k-fold cross-validation dataloaders.
    #
    # for each fold, it yields a training loader, validation loader, test loader,
    # and the positive class weight calculated *only* on that fold's training data.
    if seed is None:
        seed = config.random_seed
        
    print(f"seeding stratifiedkfold with seed: {seed}")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- preparing fold {fold + 1}/{num_folds} ---")
        
        # 1. split data into a training/validation set and a final test set for this fold
        X_train_val, X_test = X_full[train_val_idx], X_full[test_idx]
        y_train_val, y_test = y_full[train_val_idx], y_full[test_idx]
        
        # 2. adjust validation size relative to the new training/validation set size
        # this ensures the validation set is a fraction of the train_val data, not the original full data
        val_size_adj = config.val_size / (1.0 - (1.0 / num_folds))

        # 3. sub-split the train_val set into the final training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size_adj, 
            random_state=config.random_seed, 
            stratify=y_train_val
        )
        
        # 4. compute positive class weight *only* on the training data for this fold
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
        pos_weight_fold = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        print(f"positive class weight for fold {fold + 1}: {pos_weight_fold.item():.3f}")
        
        # 5. create dataset objects for this fold
        train_dataset = ProstateCancerDataset(X_train, y_train)
        val_dataset = ProstateCancerDataset(X_val, y_val)
        test_dataset = ProstateCancerDataset(X_test, y_test)

        # 6. create dataloader objects for this fold
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        print(f"fold {fold + 1} sizes: train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
        print(f"fold {fold + 1} batches: train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
        
        # 7. yield the loaders and the fold-specific weight
        yield train_loader, val_loader, test_loader, pos_weight_fold

