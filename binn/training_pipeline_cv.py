import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch_geometric.loader import DataLoader

import wandb

from config.gnn_config import GNNConfig
from config.binn_config import BinnConfig
from config.gnn_binn_config import GNNBinnConfig

from binn.src.model.create_neural_net import create_pathway_network, create_binn
from binn.src.data_loader import load_initial_data, create_cv_dataloaders, load_full_aligned_data
from binn.src.training_functions import EarlyStopping, setup_wandb, train_epoch, validate_epoch


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
    # calculates the mean and std of metrics across all folds.
    df = pd.DataFrame(fold_metrics_list)
    mean_metrics = df.mean().to_dict()
    std_metrics = df.std().to_dict()
    
    # rename keys for clarity
    summary = {}
    for key, value in mean_metrics.items():
        summary[f"{key}_mean"] = value
    for key, value in std_metrics.items():
        summary[f"{key}_std"] = value
        
    return summary


def main():
    binn_config = BinnConfig()

    # set random seed for reproducibility
    set_random_seed(binn_config.random_seed)

    # load initial data to find available features
    data, features, available_genes_in_data = load_initial_data(binn_config)
    

    # create neural network
    input_dim, hidden_dims, output_dim, binn_edge_index, pathway_net = create_pathway_network(binn_config,
                                                                                              available_features_in_data=available_genes_in_data)


    # load/align full data (adapted from your create_aligned_dataloaders, but no splits)
    X_full, y_full, pos_weight_full, network_ordered_protein_ids = load_full_aligned_data(
        binn_config, pathway_net, data, features
    )

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

    # create the k-fold dataloader generator (yields per fold: train_loader, val_loader, test_loader, pos_w)
    kfold_dataloaders = create_cv_dataloaders(
        X_full, y_full, config=binn_config, num_folds=binn_config.num_folds
    )

    for fold, (train_loader, val_loader, test_loader, pos_w) in enumerate(kfold_dataloaders):
        pos_w = pos_w.to(device)
        print(f"\n--- fold {fold + 1} / {binn_config.num_folds} ---")
        # setup wandb for the current fold
        run = setup_wandb(binn_config, device,
                          group=wandb_group_name, job_type=f'fold_{fold+1}')
        # initialize binn model
        binn_model = create_binn(input_dim=input_dim,
                                hidden_dims=hidden_dims,
                                output_dim=output_dim,
                                edge_index=binn_edge_index)
        
        binn_model.to(device)
        print(binn_model)

        # initialize the optimizer
        optimizer = torch.optim.SGD(binn_model.parameters(),
                           lr=binn_config.initial_lr,
                           momentum=binn_config.momentum,
                           weight_decay=binn_config.weight_decay,
                           nesterov=True)
        
        # initialize the learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=binn_config.num_epochs, eta_min=binn_config.min_lr)
        
        # initialize the loss function
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_val_auc = 0.0
        best_model_path_fold = os.path.join(binn_config.model_save_path, f"{binn_config.model_name}_fold_{fold+1}.pth")
       
        best_acc = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_auprc = 0.0
        best_epoch = 0  # track per fold

        for epoch in range(binn_config.num_epochs):
            print(f"epoch {epoch + 1}/{binn_config.num_epochs}")

            # training phase
            train_loss, train_metrics, binn_model, train_loader, criterion, optimizer, device, epoch = train_epoch(
                binn_model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
            )

            # validation phase
            val_loss, val_metrics, binn_model, val_loader, criterion, device, epoch = validate_epoch(
                binn_model,
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
                torch.save(binn_model.state_dict(), best_model_path_fold)

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
        binn_model.load_state_dict(torch.load(best_model_path_fold))
        test_loss, test_metrics, binn_model, test_loader, criterion, device, epoch = validate_epoch(
            binn_model,
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

        # log test results for this fold to wandb
        wandb.log({f"fold_test/{k}": v for k, v in test_metrics.items()})
        all_fold_test_metrics.append(test_metrics)
        
        # finish the wandb run for the current fold
        run.finish()

    # final aggregation and logging
    aggregated_results = aggregate_fold_results(all_fold_test_metrics)

    # print final aggregated results
    print("\n===== pure k-fold cross-validation results =====")
    for key, value in aggregated_results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()