import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from collections import defaultdict

import wandb


from config.gnn_config import GNNConfig
from config.binn_config import BinnConfig
from config.gnn_binn_config import GNNBinnConfig

from binn.src.model.create_neural_net import create_pathway_network, create_binn
from binn.src.data_loader import load_initial_data
from binn.src.training_functions import EarlyStopping

from gnn_binn.src.string_files_preprocess import map_proteins_to_genes
from gnn_binn.src.data_loader import GNNProstateCancerDataset, load_full_aligned_data, create_cv_dataloaders_with_id
from gnn_binn.src.model.gnn_binn import GNN_BINN
from gnn_binn.src.training_functions import setup_wandb, train_epoch, validate_epoch
from gnn_binn.src.explainer import  normalize_scores_with_id, explain_model_with_id


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_and_plot_contributions(model, model_path, test_loader, device, fold_info, save_path=None): 
    # analyzes the contribution of each prediction head on the test set, creates a box plot,
 
    # load the best model weights and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_head_predictions = []

    # run inference on the test set
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"analyzing head contributions for {fold_info}"):
            data = data.to(device)
            _, layer_preds = model(data, return_layer_preds=True) 
            
            if layer_preds is not None:
                all_head_predictions.append(layer_preds.t().cpu().numpy())

    if not all_head_predictions:
        print(f"warning: no layer predictions were collected for {fold_info}. cannot generate plot.")
        return

    # format data for plotting
    all_head_predictions = np.vstack(all_head_predictions)
    num_hidden_layers = all_head_predictions.shape[1]
    layer_names = [f'head {i+1}' for i in range(num_hidden_layers)]
    df = pd.DataFrame(all_head_predictions, columns=layer_names)

    # generate the box plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.boxplot(data=df, ax=ax, palette="viridis", showfliers=False)
    
    ax.set_ylabel('raw logit value', fontsize=12)
    ax.axhline(0, ls='--', color='gray', linewidth=1.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # save the plot if a path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        filename_info = fold_info.replace(" ", "_").replace(",", "")
        plot_filename = f"head_contributions_{filename_info}.png"
        full_save_path = os.path.join(save_path, plot_filename)
        
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"plot saved to: {full_save_path}")

    # isplay the plot and clean up
    print(f"displaying analysis plot for {fold_info}...")
    plt.show()
    plt.close(fig) 



def set_random_seed(seed=42):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aggregate_fold_results(fold_metrics_list):
    df = pd.DataFrame(fold_metrics_list)
    mean_metrics = df.mean().to_dict()
    std_metrics = df.std().to_dict()
    
    summary = {}
    for key, value in mean_metrics.items():
        summary[f"{key}_mean"] = value
    for key, value in std_metrics.items():
        summary[f"{key}_std"] = value
        
    return summary


def retrain_final_model(X_full, y_full, edge_index, num_nodes, input_features, binn_edge_index, input_dim, hidden_dims, output_dim,
                        gnn_binn_config, binn_config, device, avg_best_epoch, pos_weight_full):
    print("\n--- final retraining on full data ---")
    print(f"using average best epoch from cv: {avg_best_epoch}")
    
    setup_wandb(gnn_binn_config, binn_config, device)
    
    full_dataset = GNNProstateCancerDataset(X_full, y_full, edge_index, num_nodes, input_features)
    full_loader = DataLoader(full_dataset, batch_size=gnn_binn_config.batch_size, shuffle=True)
    
    pos_w = pos_weight_full.to(device)
    
    binn_model = create_binn(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, edge_index=binn_edge_index)
    gnn_binn_model = GNN_BINN(
        gnn_input_features=input_features,
        gnn_hidden_dim=gnn_binn_config.hidden_dim,
        gnn_output_features=input_features,
        binn_model=binn_model,
        num_nodes=num_nodes,
        dropout_rate=gnn_binn_config.dropout_rate,
        aggr_method=gnn_binn_config.aggregation_method
    )
    gnn_binn_model.to(device)
    
    optimizer = torch.optim.Adam(gnn_binn_model.parameters(), lr=gnn_binn_config.initial_lr, weight_decay=gnn_binn_config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=avg_best_epoch, eta_min=gnn_binn_config.min_lr)  # adjust t_max to avg_epoch
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    for epoch in range(avg_best_epoch):
        print(f"epoch {epoch + 1}/{avg_best_epoch}")
        train_loss, train_metrics, gnn_binn_model, train_loader, criterion, optimizer, device, epoch = train_epoch(
            gnn_binn_model, full_loader, criterion, optimizer, device, epoch
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "final/train/loss": train_loss,
            "final/train/acc": train_metrics["accuracy"],
            "final/train/precision": train_metrics["precision"],
            "final/train/recall": train_metrics["recall"],
            "final/train/f1": train_metrics["f1"],
            "final/train/auc": train_metrics["auc"],
            "final/train/auprc": train_metrics["auprc"],
            "final/epoch": epoch + 1,
            "final/lr": current_lr
        })
    
    final_model_path = os.path.join(gnn_binn_config.model_save_path, f"{gnn_binn_config.model_name}_final.pth")
    torch.save(gnn_binn_model.state_dict(), final_model_path)
    print(f"final model saved at {final_model_path} for interpretation.")



def main():
    gnn_config = GNNConfig()
    binn_config = BinnConfig()
    gnn_binn_config = GNNBinnConfig()

    set_random_seed(gnn_binn_config.random_seed)

    # load initial data to find available features
    data, features, available_genes_in_data = load_initial_data(binn_config)

    # map proteins in string to uniprot gene ids 
    available_genes_in_string = map_proteins_to_genes(
                    gnn_config.protein_nodes_path,
                    gnn_config.ppi_edges_path,
                    gnn_config.mapping_file_path,
                    gnn_config.gene_nodes_path,
                    gnn_config.ggi_edges_path
                )
    

    # create neural network
    input_dim, hidden_dims, output_dim, binn_edge_index, pathway_net = create_pathway_network(binn_config,
                                                                                              available_features_in_data=available_genes_in_data,
                                                                                              available_nodes_in_string=available_genes_in_string)


    # load/align full data 
    X_full, y_full, edge_index, pos_weight_full, network_ordered_protein_ids = load_full_aligned_data(
        binn_config, gnn_binn_config, pathway_net, data, features
    )

    # save the correctly ordered y_full array for the clustering script
    y_full_save_path = os.path.join(gnn_binn_config.scores_dir, 'y_full_ordered.npy')
    np.save(y_full_save_path, y_full)
    print(f"saved correctly ordered labels for clustering to {y_full_save_path}")

    # model initialization
    input_features = 2 if binn_config.data_type == 'combined' else 1
    num_nodes = len(network_ordered_protein_ids)

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # k-fold cross-validation
    all_fold_test_metrics = []
    best_epochs = []

    # unique group name for this entire k-fold experiment run
    wandb_group_name = f"exp_{wandb.util.generate_id()}"

    # structure: {layer_name: {sample_idx: [score_array_rep1, score_array_rep2, ...]}}
    all_sample_scores = defaultdict(lambda: defaultdict(list))


    for i in range(gnn_binn_config.n_repeats):
        print(f"\n{'#'*25} repetition {i + 1} / {gnn_binn_config.n_repeats} {'#'*25}")

        # set a new seed for each repetition to ensure different data splits
        repetition_seed = gnn_binn_config.random_seed + i
        set_random_seed(repetition_seed)

        # create new folds for the current repetition
        print(f"creating new cv folds for repetition {i+1}...")

        # create the k-fold dataloader generator (yields per fold: train_loader, val_loader, test_loader, pos_w)
        kfold_dataloaders = create_cv_dataloaders_with_id(
            X_full, y_full, edge_index, num_nodes, input_features, config=gnn_binn_config, seed=repetition_seed
        )

        for fold, (train_loader, val_loader, test_loader, pos_w, test_indices) in enumerate(kfold_dataloaders):
            pos_w = pos_w.to(device)
            print(f"\n--- fold {fold + 1} / {gnn_binn_config.num_folds} for repetition {i + 1} ---")
            # setup wandb for the current fold
            run = setup_wandb(gnn_binn_config, binn_config, device,
                            group=wandb_group_name, job_type=f'fold_{fold+1}')
            # initialize binn model
            binn_model = create_binn(input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim,
                                    edge_index=binn_edge_index)
            
            # initialize the model
            gnn_binn_model = GNN_BINN(
                gnn_input_features=input_features,
                gnn_hidden_dim=gnn_binn_config.hidden_dim,
                gnn_output_features=input_features,
                binn_model=binn_model,
                num_nodes=num_nodes,
                dropout_rate=gnn_binn_config.dropout_rate,
                aggr_method=gnn_binn_config.aggregation_method
            )
            gnn_binn_model.to(device)
            print(gnn_binn_model)

            # initialize the optimizer
            optimizer = torch.optim.Adam(gnn_binn_model.parameters(),
                                        lr=gnn_binn_config.initial_lr,
                                        weight_decay=gnn_binn_config.weight_decay)
            
            # optimizer = torch.optim.sgd(gnn_binn_model.parameters(),
            #                             lr=gnn_binn_config.initial_lr,
            #                             momentum=gnn_binn_config.momentum, 
            #                             weight_decay=gnn_binn_config.weight_decay,  
            #                             nesterov=true)
            
            # initialize the learning rate scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=gnn_binn_config.num_epochs, eta_min=gnn_binn_config.min_lr)
            
            # initialize the loss function
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

            best_val_auc = 0.0
            best_model_path_fold = os.path.join(gnn_binn_config.model_save_path, \
            f"{gnn_binn_config.model_name}_repeat_{i+1}_fold_{fold+1}.pth")

            best_acc = 0.0
            best_precision = 0.0
            best_recall = 0.0
            best_f1 = 0.0
            best_auprc = 0.0
            best_epoch = 0  

            for epoch in range(gnn_binn_config.num_epochs):
                print(f"epoch {epoch + 1}/{gnn_binn_config.num_epochs}")

                # training phase
                train_loss, train_metrics, gnn_binn_model, train_loader, criterion, optimizer, device, epoch = train_epoch(
                    gnn_binn_model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    epoch,
                )

                # validation phase
                val_loss, val_metrics, gnn_binn_model, val_loader, _, _ = validate_epoch(
                    gnn_binn_model,
                    val_loader,
                    criterion,
                    device,
                    epoch,
                )

                # step the scheduler
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # log metrics to wandb
                wandb.log({
                    "epoch/train/loss": train_loss,
                    "epoch/train/acc": train_metrics["accuracy"],
                    "epoch/train/precision": train_metrics["precision"],
                    "epoch/train/recall": train_metrics["recall"],
                    "epoch/train/f1": train_metrics["f1"],
                    "epoch/train/auc": train_metrics["auc"],
                    "epoch/train/auprc": train_metrics["auprc"],
                    "epoch/val/loss": val_loss,
                    "epoch/val/acc": val_metrics["accuracy"],
                    "epoch/val/precision": val_metrics["precision"],
                    "epoch/val/recall": val_metrics["recall"],
                    "epoch/val/f1": val_metrics["f1"],
                    "epoch/val/auc": val_metrics["auc"],
                    "epoch/val/auprc": val_metrics["auprc"],
                    "epoch": epoch + 1,
                    "epoch/lr": current_lr
                })
                if val_metrics["auc"] > best_val_auc:
                    best_val_auc = val_metrics["auc"]
                    best_acc = val_metrics["accuracy"]
                    best_precision = val_metrics["precision"]
                    best_recall = val_metrics["recall"]
                    best_f1 = val_metrics["f1"]
                    best_auprc = val_metrics["auprc"]
                    best_epoch = epoch + 1
                    # save the model for this specific fold
                    torch.save(gnn_binn_model.state_dict(), best_model_path_fold)

            best_epochs.append(best_epoch)
            print(f"fold {fold + 1} training complete. best epoch: {best_epoch}")

            # print results for this fold (best val)
            print("\n===== best val model evaluation =====")
            print(f"accuracy: {best_acc:.4f}")
            print(f"precision: {best_precision:.4f}")
            print(f"recall: {best_recall:.4f}")
            print(f"f1 score: {best_f1:.4f}")
            print(f"auc: {best_val_auc:.4f}")
            print(f"auprc: {best_auprc:.4f}")

            # evaluate the best model on the fold's test set
            gnn_binn_model.load_state_dict(torch.load(best_model_path_fold))
            test_loss, test_metrics, gnn_binn_model, test_loader, _, _ = validate_epoch(
                gnn_binn_model,
                test_loader,
                criterion,
                device,
            )

            # print test metrics
            print("\n===== fold test set evaluation =====")
            print(f"test loss: {test_loss:.4f}")
            print(f"test accuracy: {test_metrics['accuracy']:.4f}")
            print(f"test precision: {test_metrics['precision']:.4f}")
            print(f"test recall: {test_metrics['recall']:.4f}")
            print(f"test f1 score: {test_metrics['f1']:.4f}")
            print(f"test auc: {test_metrics['auc']:.4f}")
            print(f"test auprc: {test_metrics['auprc']:.4f}")

            print(f"\n--- analyzing prediction head contributions for repetition {i+1}, fold {fold+1} ---")
            analyze_and_plot_contributions(
                model=gnn_binn_model,
                model_path=best_model_path_fold,
                test_loader=test_loader,
                device=device,
                fold_info=f"repetition {i + 1}, fold {fold + 1}",
                save_path=gnn_binn_config.plot_save_path
            )

            # log test results for this fold to wandb
            wandb.log({f"fold_test/{k}": v for k, v in test_metrics.items()})
            all_fold_test_metrics.append(test_metrics)
            
            # finish the wandb run for the current fold
            run.finish()

            print(f"\n--- explaining model for repetition {i+1}, fold {fold+1} ---")
            gnn_binn_model.eval()
            binn_model = gnn_binn_model.binn

            per_sample_scores_fold = explain_model_with_id(
                gnn_binn_model=gnn_binn_model,
                binn_model=binn_model,
                test_loader=test_loader, # explain on the test set for this fold
                device=device,
                hidden_dims=hidden_dims,
                num_runs=gnn_binn_config.num_runs,
                n_steps=gnn_binn_config.n_steps,
                save_path=none 
            )

            print(f"--- normalizing explanation scores for fold {fold+1} ---")

            normalized_scores_fold = normalize_scores_with_id(
                scores=per_sample_scores_fold,
                model=binn_model,
                method=gnn_binn_config.normalization_method 
            )

            # populate the master data structure
            for layer_name, scores_numpy in normalized_scores_fold.items():
                
                # iterate through the samples of this test set
                for j, original_sample_idx in enumerate(test_indices):
                    # get the score vector for the j-th sample in this test set
                    sample_score_vector = scores_numpy[j, :]
                    
                    # append it to the list for that specific global sample index
                    all_sample_scores[layer_name][original_sample_idx].append(sample_score_vector)
            
            print("--- explanation complete and scores stored. ---")

    # final aggregation and logging
    aggregated_results = aggregate_fold_results(all_fold_test_metrics)

    # print final aggregated results
    print("\n===== pure k-fold cross-validation results =====")
    for key, value in aggregated_results.items():
        print(f"{key}: {value:.4f}")



    print("analyzing interpretation stability across all runs")

    final_stability_results = []

    for layer_name, sample_data in all_sample_scores.items():
        # get the number of nodes/features for this layer from the first sample's data
        num_nodes_in_layer = sample_data[0][0].shape[0]
        
        # this matrix will hold the final, stable score for each sample for each node
        # the score is the average across repetitions
        stable_sample_scores = np.zeros((len(x_full), num_nodes_in_layer))

        for sample_idx, score_list in sample_data.items():
            # score_list contains n_repeats arrays for this sample
            # stack them and average across repetitions to get one stable score vector
            stable_score_vector = np.mean(np.stack(score_list), axis=0)
            stable_sample_scores[sample_idx, :] = stable_score_vector

        # calculate the final mean and std across all samples
        # we use absolute values for importance magnitude
        mean_importance_per_node = np.mean(np.abs(stable_sample_scores), axis=0)
        std_importance_per_node = np.std(np.abs(stable_sample_scores), axis=0)

        # determine the corresponding layer index in the pathway_net object
        num_pathway_layers = len(pathway_net.layer_indices) - 1
        protein_layer_idx = num_pathway_layers - 1 

        pathway_layer_idx = -1 # initialize to an invalid value

        if layer_name == 'end_to_end':
            # attributions are on the inputs to the binn -> the protein layer
            pathway_layer_idx = protein_layer_idx
        else:
            # 'hidden_0' attributions are on the inputs to hidden_0, which are the proteins.
            # 'hidden_1' attributions are on the inputs to hidden_1, which are the nodes of hidden_0.
            # therefore, the mapping for hidden_idx is to the layer that feeds it.
            hidden_idx = int(layer_name.split('_')[1])
            pathway_layer_idx = protein_layer_idx - hidden_idx
        
        # check for a valid mapping
        if pathway_layer_idx < 0:
            print(f"warning: could not map binn layer '{layer_name}' to a valid pathway_net layer (index went below 0). skipping.")
            continue
        
        # get the global starting index for this layer from pathway_net
        global_start_index = pathway_net.layer_indices[pathway_layer_idx]

        # collect node names for this layer (for saving stable_sample_scores)
        node_names = []
        for node_idx in range(num_nodes_in_layer):
            global_node_index = global_start_index + node_idx
            node_id = pathway_net.index_to_node.get(global_node_index, f"unknownid_{global_node_index}")
            node_name = pathway_net.id_to_name.get(node_id, node_id)
            node_names.append(node_name)

        # create and save the per-sample scores dataframe
        sample_ids = np.arange(len(x_full))
        sample_scores_df = pd.DataFrame(stable_sample_scores, index=sample_ids, columns=node_names)
        save_path = os.path.join(gnn_binn_config.scores_dir, f"stable_sample_scores_{layer_name}.csv")
        sample_scores_df.to_csv(save_path, index_label="sample_id")
        print(f"saved per-sample stable scores for layer '{layer_name}' to {save_path}")

        # loop through the nodes in this layer and perform the name lookup for final_stability_results
        for node_idx in range(num_nodes_in_layer):
            global_node_index = global_start_index + node_idx
            
            node_id = pathway_net.index_to_node.get(global_node_index, f"unknownid_{global_node_index}")
            node_name = pathway_net.id_to_name.get(node_id, node_id) 

            final_stability_results.append({
                "layer": layer_name,
                "node_name": node_name,
                "mean_importance": mean_importance_per_node[node_idx],
                "std_dev_importance": std_importance_per_node[node_idx],
            })

    # convert to dataframe and save
    final_df = pd.dataframe(final_stability_results)
    final_df = final_df.sort_values(by=["layer", "mean_importance"], ascending=[true, false])
    
    output_path = os.path.join(gnn_binn_config.scores_dir, "final_interpretation_stability_results.csv")
    final_df.to_csv(output_path, index=false)
    
    print(f"\nsuccessfully saved final stability results to: {output_path}")
    

    print("\n" + "="*50)
    print(f"top {gnn_binn_config.top_k} most important nodes per layer")
    print("="*50)

    # group the dataframe by the 'layer' column
    for layer_name, group_df in final_df.groupby('layer'):
        print(f"\n--- top {gnn_binn_config.top_k} nodes for layer: {layer_name} ---")
        # get the top k rows from this group (which is already sorted by importance)
        top_k_nodes = group_df.head(gnn_binn_config.top_k)
        print(top_k_nodes.to_string(index=false))
    


if __name__ == "__main__":
    main()