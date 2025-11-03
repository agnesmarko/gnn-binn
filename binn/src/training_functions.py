from tqdm import tqdm
import torch
import wandb
import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='max', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1


    def __call__(self, current_score, model, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
            return False
        
        # check if current score is better than the best score
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Restored weights from epoch {self.best_epoch}")
                return True
            
            return False
            
        
def setup_wandb(config, device, group=None, job_type=None):
    run = wandb.init(
        project=config.wandb_project,

        group=group,
        job_type=job_type,
        reinit=True,

        config={
            # Reproducibility
            "random_seed": config.random_seed,

            # Model parameters
            "normalization": config.norm,
            "weight_initialization": config.weight_init,
            "root_nodes_included": config.root_nodes_to_include,
            "max_level": config.max_level,

            # Training parameters
            "batch_size": config.batch_size,
            "initial_lr": config.initial_lr,
            "min_lr": config.min_lr,
            "weight_decay": config.weight_decay,
            "num_epochs": config.num_epochs,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "loss_function": config.loss_function,
            "momentum": config.momentum,

            # Dataset parameters
            "test_size": config.test_size,
            "data_type": config.data_type,

            # Early stopping parameters
            "patience": config.patience,
            "min_delta": config.min_delta,

            # Device
            "device": str(device),
            "num_workers": config.num_workers
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


def train_epoch_old(model, dataloader, criterion, optimizer, device, epoch, log_every_n_batches):
    model.train()
    running_loss = 0.0
    running_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0, 'auprc': 0.0}

    # batch running loss
    batch_running_loss = 0.0
    batch_running_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0, 'auprc': 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} training")

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # ensure the labels tensor contains floating-point numbers
        # reshape the labels tensor to have shape [batch_size, 1]
        labels = labels.float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backpropagate
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update the weights, aka tell the 'optimizer' to adjust the parameters based on the gradients computed
        # in 'loss.backward()'
        optimizer.step()

        # calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_metrics(outputs, labels)

        # update running loss and metrics
        running_loss += loss.item()
        for key in running_metrics.keys():
            running_metrics[key] += batch_metrics[key]
        
        # update batch level running loss and metrics
        batch_running_loss += loss.item()
        for key in batch_running_metrics.keys():
            batch_running_metrics[key] += batch_metrics[key]

        # update progress bar with acc, auc, auprc
        pbar.postfix = {
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_metrics['accuracy']:.4f}",
            "auc": f"{batch_metrics['auc']:.4f}",
            "auprc": f"{batch_metrics['auprc']:.4f}",
        }

        # log every n batches to wandb
        if log_every_n_batches is not None and (batch_idx + 1) % log_every_n_batches == 0:
            print(f"\nLogging at batch {batch_idx + 1}")
            print(f"Batch running loss: {batch_running_loss / log_every_n_batches:.4f}")
            # print(f"Batch accuracy: {batch_running_metrics['accuracy'] / log_every_n_batches:.4f}")
            # print(f"Batch AUC: {batch_running_metrics['auc'] / log_every_n_batches:.4f}")
            # print(f"Batch AUPRC: {batch_running_metrics['auprc'] / log_every_n_batches:.4f}")
            # print(f"Batch precision: {batch_running_metrics['precision'] / log_every_n_batches:.4f}")
            # print(f"Batch recall: {batch_running_metrics['recall'] / log_every_n_batches:.4f}")
            # print(f"Batch F1: {batch_running_metrics['f1'] / log_every_n_batches:.4f}")

            # wandb.log({
            #     "batch/train/loss": batch_running_loss / log_every_n_batches,
            #     "batch/train/acc": batch_running_metrics["accuracy"] / log_every_n_batches,
            #     "batch/train/precision": batch_running_metrics["precision"] / log_every_n_batches,
            #     "batch/train/recall": batch_running_metrics["recall"] / log_every_n_batches,
            #     "batch/train/f1": batch_running_metrics["f1"] / log_every_n_batches,
            #     "batch/train/auc": batch_running_metrics["auc"] / log_every_n_batches,
            #     "batch/train/auprc": batch_running_metrics["auprc"] / log_every_n_batches,
            # })
            batch_running_loss = 0.0
            batch_running_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0, 'auprc': 0.0}

    # epoch average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {key: value / len(dataloader) for key, value in running_metrics.items()}

    return epoch_loss, epoch_metrics, model, dataloader, criterion, optimizer, device, epoch


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_every_n_batches=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} training")

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # ensure the labels tensor contains floating-point numbers
        # reshape the labels tensor to have shape [batch_size, 1]
        labels = labels.float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backpropagate
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update the weights, aka tell the 'optimizer' to adjust the parameters based on the gradients computed
        # in 'loss.backward()'
        optimizer.step()

        # update running loss and metrics
        running_loss += loss.item()

        all_labels.append(labels)
        all_outputs.append(outputs)

        # update progress bar with acc, auc, auprc
        pbar.postfix = {
            "loss": f"{loss.item():.4f}"
        }

    # epoch average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    
    all_labels = torch.cat(all_labels, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    epoch_metrics = calculate_metrics(all_outputs, all_labels)

    # print epoch metrics (acc, prec, rec, f1, auc, auprc)
    print(f"acc: {epoch_metrics['accuracy']:.4f}, prec: {epoch_metrics['precision']:.4f}, \
          rec: {epoch_metrics['recall']:.4f}, f1: {epoch_metrics['f1']:.4f}, auc: {epoch_metrics['auc']:.4f}, \
            auprc: {epoch_metrics['auprc']:.4f}")


    return epoch_loss, epoch_metrics, model, dataloader, criterion, optimizer


def validate_epoch(model, dataloader, criterion, device, epoch=None):
    all_labels = []
    all_outputs = []

    model.eval()
    running_loss = 0.0

    if epoch is not None:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} validation")
    else:
        pbar = tqdm(dataloader, desc=f"Test evaluation")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # ensure the labels tensor contains floating-point numbers
            # reshape the labels tensor to have shape [batch_size, 1]
            labels = labels.float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            all_labels.append(labels)
            all_outputs.append(outputs)

            # update running loss 
            running_loss += loss.item()

            pbar.postfix = {
                "loss": f"{loss.item():.4f}"
            }
            

            
    # average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    all_labels = torch.cat(all_labels, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_probs = torch.sigmoid(all_outputs)


    epoch_metrics = calculate_metrics(all_outputs, all_labels)

    y_true_np = all_labels.cpu().numpy()
    y_pred_proba_np = all_probs.cpu().numpy()

    # print epoch metrics (acc, prec, rec, f1, auc, auprc)
    print(f"acc: {epoch_metrics['accuracy']:.4f}, prec: {epoch_metrics['precision']:.4f}, \
          rec: {epoch_metrics['recall']:.4f}, f1: {epoch_metrics['f1']:.4f}, auc: {epoch_metrics['auc']:.4f}, \
            auprc: {epoch_metrics['auprc']:.4f}")

    return epoch_loss, epoch_metrics, model, dataloader, y_true_np, y_pred_proba_np