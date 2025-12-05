import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
import wandb



def setup_wandb(gnn_binn_config, binn_config, device, group=None, job_type=None):
    run = wandb.init(
        project=gnn_binn_config.wandb_project,

        group=group,
        job_type=job_type,
        reinit=True,

        config={
            # Data
            "data_type": binn_config.data_type,

            # Reproducibility
            "random_seed": gnn_binn_config.random_seed,

            # Model parameters
            "hidden_dim": gnn_binn_config.hidden_dim,
            "dropout": gnn_binn_config.dropout_rate,
            "aggr": gnn_binn_config.aggregation_method,

            # Training parameters
            "batch_size": gnn_binn_config.batch_size,
            "initial_lr": gnn_binn_config.initial_lr,
            "min_lr": gnn_binn_config.min_lr,
            "weight_decay": gnn_binn_config.weight_decay,
            "num_epochs": gnn_binn_config.num_epochs,
            "momentum": gnn_binn_config.momentum,
            "optimizer": gnn_binn_config.optimizer,
            "warmup_epochs": gnn_binn_config.warmup_epochs,
            "scheduler": gnn_binn_config.scheduler,
            "loss_function": gnn_binn_config.loss_function,

            # Dataset parameters
            "test_size": gnn_binn_config.test_size,
            "val_size": gnn_binn_config.val_size,

            # Early stopping parameters
            "patience": gnn_binn_config.patience,
            "min_delta": gnn_binn_config.min_delta,

            # Device
            "device": str(device),
            "num_workers": gnn_binn_config.num_workers
        }
    )

    return run


def calculate_metrics(outputs, labels, threshold=0.5):
    # the 'outputs' are now raw logits
    # need to convert them to probabilities
    labels = labels.detach().cpu().numpy().flatten()

    probabilities = torch.sigmoid(outputs).detach().cpu().numpy().flatten()

    preds = (probabilities > threshold).astype(float)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities)
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} training")

    for data_batch in pbar:
        data_batch = data_batch.to(device)
        labels = data_batch.y.float().view(-1, 1)

        optimizer.zero_grad()
        
        # Combined model forward pass
        outputs = model(data_batch)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        
        all_labels.append(labels)
        all_outputs.append(outputs)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    
    # Calculate metrics for the entire epoch
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    epoch_metrics = calculate_metrics(all_outputs, all_labels)

    # print epoch metrics (acc, auc, auprc)
    print(f"acc: {epoch_metrics['accuracy']:.4f}, auc: {epoch_metrics['auc']:.4f}, auprc: {epoch_metrics['auprc']:.4f}")
    
    return epoch_loss, epoch_metrics, model, dataloader, criterion, optimizer, device, epoch


def validate_epoch(model, dataloader, criterion, device, epoch=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    desc = f"Epoch {epoch} validation" if epoch is not None else "Test evaluation"
    pbar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for data_batch in pbar:
            data_batch = data_batch.to(device)
            labels = data_batch.y.float().view(-1, 1)

            # Combined model forward pass
            outputs = model(data_batch)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_labels.append(labels)
            all_outputs.append(outputs)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.sigmoid(all_outputs)

    epoch_metrics = calculate_metrics(all_outputs, all_labels)

    y_true_np = all_labels.cpu().numpy()
    y_pred_proba_np = all_probs.cpu().numpy()

    print(f"acc: {epoch_metrics['accuracy']:.4f}, auc: {epoch_metrics['auc']:.4f}, auprc: {epoch_metrics['auprc']:.4f}")

    return epoch_loss, epoch_metrics, model, dataloader, y_true_np, y_pred_proba_np