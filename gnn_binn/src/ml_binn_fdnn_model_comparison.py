import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as GNNDataLoader
from torch.utils.data import DataLoader as TorchDataLoader # For standard BINN

import wandb
import warnings
warnings.filterwarnings('ignore')

# import torch data loader  

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
sys.path.append(os.path.join(script_dir, '../..'))


from config.binn_config import BinnConfig
from config.gnn_config import GNNConfig
from config.gnn_binn_config import GNNBinnConfig

from binn.src.model.create_neural_net import create_pathway_network, create_binn
from gnn_binn.src.string_files_preprocess import map_proteins_to_genes
from gnn_binn.src.model.gnn_binn import GNN_BINN

# GNN-BINN specific loaders
from gnn_binn.src.data_loader import (create_cv_dataloaders as create_gnn_cv_dataloaders,
                                      load_full_aligned_data as load_gnn_data)
# Standard BINN specific loaders
from binn.src.data_loader import (load_initial_data,
                                   create_cv_dataloaders as create_binn_cv_dataloaders,
                                   load_full_aligned_data as load_binn_data)

from gnn_binn.src.training_functions import (setup_wandb as setup_wandb_gnn_binn,
                                             train_epoch as train_gnn_binn_epoch,
                                             validate_epoch as validate_gnn_binn_epoch)

from binn.src.training_functions import (setup_wandb as setup_wandb_binn,
                                         train_epoch as train_binn_epoch,
                                         validate_epoch as validate_binn_epoch)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve


from scipy.stats import ttest_rel

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class FDNN(nn.Module):
    """
    Fully Connected Dense Neural Network for binary classification.
    Architecture: 7419 -> 2137 -> 2319 -> 2401 -> 2240 -> 1389 -> 719 -> 195 -> 43 -> 1
    """
    def __init__(self, input_dim=7419, dropout_rate=0.3):
        super(FDNN, self).__init__()

        # Define layer dimensions
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 2137),    # Hidden 1
            nn.Linear(2137, 2319),          # Hidden 2
            nn.Linear(2319, 2401),          # Hidden 3
            nn.Linear(2401, 2240),          # Hidden 4
            nn.Linear(2240, 1389),          # Hidden 5
            nn.Linear(1389, 719),           # Hidden 6
            nn.Linear(719, 195),            # Hidden 7
            nn.Linear(195, 43),             # Hidden 8
            nn.Linear(43, 1)                # Output
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        return x.squeeze(-1)  



def plot_comparison_curves(self, save_path=None):
        """
        Generates and saves model comparison ROC and AUPRC curves.

        Args:
            save_path (str, optional): Directory to save the plots. If None, plots are only shown.
                                       Defaults to None.
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"Plots will be saved to: {os.path.abspath(save_path)}")

        sns.set_style("white")
        custom_colors = [
            '#457B9D',  # GNN-BINN
            '#1A7431',  # BINN
            '#9B59B6',  # FDNN 
            '#FFF3B0',  # Decision Tree
            '#E09F3E',  # Logistic Regression
            '#9E2A2B'   # Random Forest
        ]

        fig_roc, ax_roc = plt.subplots(figsize=(9, 9))

        for i, (model_name, results) in enumerate(self.plot_data.items()):
            if not results: continue
            tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
            for fold_res in results:
                fpr, tpr, _ = roc_curve(fold_res['y_true'], fold_res['y_pred'])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(auc(fpr, tpr))
            
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc, std_auc = np.mean(aucs), np.std(aucs)
            # Use the custom color list
            ax_roc.plot(mean_fpr, mean_tpr, color=custom_colors[i],
                        label=f'{model_name}: {mean_auc:.2f} \u00B1 {std_auc:.2f}', lw=1.25)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            # Use the custom color list
            ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color=custom_colors[i], alpha=0.1)

        ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.5)
        ax_roc.set(xlabel="1-specificity", ylabel="Sensitivity")
        ax_roc.legend(loc="lower right", title="AUC-ROC", fontsize=13)
        sns.despine(ax=ax_roc, trim=True)
        if save_path:
            fig_roc.savefig(os.path.join(save_path, "model_comparison_roc_curve.png"), dpi=300)

        fig_prc, ax_prc = plt.subplots(figsize=(9, 9))
        for i, (model_name, results) in enumerate(self.plot_data.items()):
            if not results: continue
            precisions, auprcs, mean_recall = [], [], np.linspace(0, 1, 100)
            for fold_res in results:
                precision, recall, _ = precision_recall_curve(fold_res['y_true'], fold_res['y_pred'])
                precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
                auprcs.append(average_precision_score(fold_res['y_true'], fold_res['y_pred']))

            mean_precision = np.mean(precisions, axis=0)
            mean_auprc, std_auprc = np.mean(auprcs), np.std(auprcs)
            # Use the custom color list
            ax_prc.plot(mean_recall, mean_precision, color=custom_colors[i],
                        label=f'{model_name}: {mean_auprc:.2f} \u00B1 {std_auprc:.2f}', lw=1.25)
            std_precision = np.std(precisions, axis=0)
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            # Use the custom color list
            ax_prc.fill_between(mean_recall, precision_lower, precision_upper, color=custom_colors[i], alpha=0.1)
        
        ax_prc.set(xlabel="Recall", ylabel="Precision")
        ax_prc.legend(loc="lower left", title="AUPRC", fontsize=13)
        sns.despine(ax=ax_prc, trim=True)
        if save_path:
            fig_prc.savefig(os.path.join(save_path, "model_comparison_auprc_curve.png"), dpi=300)
        plt.show()


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelComparison:
    def __init__(self, binn_config: BinnConfig, gnn_config: GNNConfig, gnn_binn_config: GNNBinnConfig):
        self.binn_config = binn_config
        self.gnn_config = gnn_config
        self.gnn_binn_config = gnn_binn_config
        self.random_seed = gnn_binn_config.random_seed

        self.n_repeats = gnn_binn_config.n_repeats

        self.plot_data = {
            'GNN-BINN': [],
            'BINN': [],
            'FDNN': [],  # Add this line
            'Decision Tree': [],
            'Logistic Regression': [],
            'Random Forest': []
        }

    def prepare_initial_data(self):
        print("Preparing initial steps for data loading...")
        data, features, available_genes_in_data = load_initial_data(self.binn_config)
        available_genes_in_string = map_proteins_to_genes(
            self.gnn_config.protein_nodes_path, self.gnn_config.ppi_edges_path,
            self.gnn_config.mapping_file_path, self.gnn_config.gene_nodes_path,
            self.gnn_config.ggi_edges_path)
        
        input_dim, hidden_dims, output_dim, binn_edge_index, pathway_net = create_pathway_network(
            self.binn_config, available_genes_in_data, available_genes_in_string)
        
        return input_dim, hidden_dims, output_dim, binn_edge_index, data, features, pathway_net


    def train_fdnn_fold(self, repetition_idx, fold_idx, X_train, y_train, X_val, y_val, X_test, y_test, pos_w, device):
        """
        Trains and evaluates the FDNN model for one fold of cross-validation.
        """
        print(f"\n--- Training FDNN for Repetition {repetition_idx + 1}, Fold {fold_idx + 1} ---")

        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = TorchDataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize model
        input_dim = X_train.shape[1]
        model = FDNN(input_dim=input_dim, dropout_rate=0.3).to(device)
        print(model)
        # print number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        # Initialize optimizer, scheduler, and loss function
        pos_w = pos_w.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.binn_config.num_epochs, eta_min=1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_val_auprc = 0.0
        best_epoch = 0
        best_model_path = os.path.join(
            self.gnn_binn_config.model_save_path, 
            f"fdnn_repeat_{repetition_idx+1}_fold_{fold_idx+1}.pth"
        )

        # Training loop
        for epoch in range(self.binn_config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            train_loss /= len(train_dataset)
            train_preds = np.array(train_preds)
            train_labels = np.array(train_labels)
            train_preds_binary = (train_preds > 0.5).astype(int)


            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_x.size(0)
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())

            val_loss /= len(val_dataset)
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            val_preds_binary = (val_preds > 0.5).astype(int)

            val_metrics = {
                'accuracy': accuracy_score(val_labels, val_preds_binary),
                'precision': precision_score(val_labels, val_preds_binary, zero_division=0),
                'recall': recall_score(val_labels, val_preds_binary, zero_division=0),
                'f1': f1_score(val_labels, val_preds_binary, zero_division=0),
                'auc': roc_auc_score(val_labels, val_preds),
                'auprc': average_precision_score(val_labels, val_preds)
            }

            scheduler.step()

            # Save best model based on validation AUPRC
            if val_metrics["auprc"] > best_val_auprc:
                best_val_auprc = val_metrics["auprc"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.binn_config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val AUPRC: {val_metrics['auprc']:.4f}")

        print(f"FDNN training complete. Best epoch: {best_epoch} with Val AUPRC: {best_val_auprc:.4f}")

        # Evaluate on test set
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_preds, test_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                test_labels.extend(batch_y.cpu().numpy())

        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_preds_binary = (test_preds > 0.5).astype(int)

        test_metrics = {
            'accuracy': accuracy_score(test_labels, test_preds_binary),
            'precision': precision_score(test_labels, test_preds_binary, zero_division=0),
            'recall': recall_score(test_labels, test_preds_binary, zero_division=0),
            'f1': f1_score(test_labels, test_preds_binary, zero_division=0),
            'auc': roc_auc_score(test_labels, test_preds),
            'auprc': average_precision_score(test_labels, test_preds)
        }

        print(f"Test AUPRC: {test_metrics['auprc']:.4f}")

        return test_metrics, test_labels, test_preds


    def train_ml_model(self, model_type, X_train, y_train, X_test, y_test):
        print(f"\nTraining {model_type.upper()}...")
        if model_type == 'dt':
            model = DecisionTreeClassifier(random_state=self.random_seed)
        elif model_type == 'lr':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = LogisticRegression(random_state=self.random_seed)
        elif model_type == 'rf':
            model = RandomForestClassifier(random_state=self.random_seed)
        else:
            raise ValueError("Invalid model_type")

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auprc': average_precision_score(y_test, y_pred_proba)
        }
        print(f"{model_type.upper()} AUPRC: {metrics['auprc']:.4f}")
        return metrics, y_test, y_pred_proba


    def train_binn_fold(self, repetition_idx, fold_idx, train_loader, val_loader, test_loader, pos_w, input_dim, hidden_dims, output_dim, binn_edge_index, device, wandb_group_name):
        """
        Trains and evaluates the standard BINN model for one fold of cross-validation.
        This function now mirrors the structure of train_gnn_binn_fold.
        """
        print(f"\n--- Training BINN for Repetition {repetition_idx + 1}, Fold {fold_idx + 1} ---")
        pos_w = pos_w.to(device)
        
        # Setup wandb for the current fold
        run = setup_wandb_binn(self.binn_config, device, group=wandb_group_name, job_type=f'repeat_{repetition_idx+1}_binn_fold_{fold_idx+1}')
        
        # Initialize model
        model = create_binn(input_dim, hidden_dims, output_dim, binn_edge_index).to(device)
        print(model)

        # Initialize optimizer, scheduler, and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=self.binn_config.initial_lr, momentum=self.binn_config.momentum, weight_decay=self.binn_config.weight_decay, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.binn_config.num_epochs, eta_min=self.binn_config.min_lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_val_auc = 0.0
        best_epoch = 0
        best_model_path_fold = os.path.join(self.binn_config.model_save_path, f"{self.binn_config.model_name}_repeat_{repetition_idx+1}_fold_{fold_idx+1}.pth")

        # Training loop
        for epoch in range(self.binn_config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.binn_config.num_epochs}")

            # Training phase
            train_loss, train_metrics, _, _, _, _ = train_binn_epoch(model, train_loader, criterion, optimizer, device, epoch)

            # Validation phase
            val_loss, val_metrics, _, _, _, _ = validate_binn_epoch(model, val_loader, criterion, device, epoch)
            
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

            # Save the best model based on validation AUPRC
            if val_metrics["auprc"] > best_val_auc:
                best_val_auc = val_metrics["auprc"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path_fold)

        print(f"Repetition {repetition_idx + 1}, Fold {fold_idx + 1} BINN training complete. Best epoch: {best_epoch} with Val AUPRC: {best_val_auc:.4f}")

        # Evaluate the best model on the test set
        model.load_state_dict(torch.load(best_model_path_fold))
        test_loss, test_metrics, _, _, y_true, y_pred_proba = validate_binn_epoch(
            model, test_loader, criterion, device
        )

        print("\n===== BINN Fold Test Set Evaluation =====")
        print(f"Test AUPRC: {test_metrics['auprc']:.4f}")

        # Log test results and finish run
        wandb.log({f"fold_test/{k}": v for k, v in test_metrics.items()})
        run.finish()

        return test_metrics, y_true, y_pred_proba


    def train_gnn_binn_fold(self, repetition_idx, fold_idx, train_loader, val_loader, test_loader, pos_w, input_dim, hidden_dims, output_dim, binn_edge_index, num_nodes, input_features, device, wandb_group_name):
        print(f"\n--- Training GNN-BINN for Repetition {repetition_idx + 1}, Fold {fold_idx + 1} ---")
        pos_w = pos_w.to(device)
        run = setup_wandb_gnn_binn(self.gnn_binn_config, self.binn_config, device, group=wandb_group_name, job_type=f'repeat_{repetition_idx+1}_gnn_binn_fold_{fold_idx+1}')
        binn_model = create_binn(input_dim, hidden_dims, output_dim, binn_edge_index)
        gnn_binn_model = GNN_BINN(input_features, self.gnn_binn_config.hidden_dim, input_features, binn_model, num_nodes, self.gnn_binn_config.dropout_rate, self.gnn_binn_config.aggregation_method).to(device)
        print(gnn_binn_model)

        optimizer = torch.optim.Adam(gnn_binn_model.parameters(), lr=self.gnn_binn_config.initial_lr, weight_decay=self.gnn_binn_config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.gnn_binn_config.num_epochs, eta_min=self.gnn_binn_config.min_lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_val_auc = 0.0
        best_model_path_fold = os.path.join(self.gnn_binn_config.model_save_path, f"{self.gnn_binn_config.model_name}_repeat_{repetition_idx+1}_fold_{fold_idx+1}.pth")

        for epoch in range(self.gnn_binn_config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.gnn_binn_config.num_epochs}")
            _, _, gnn_binn_model, _, _, _, _, _ = train_gnn_binn_epoch(gnn_binn_model, train_loader, criterion, optimizer, device, epoch)
            _, val_metrics, _, _, _, _ = validate_gnn_binn_epoch(gnn_binn_model, val_loader, criterion, device, epoch)
            scheduler.step()
            if val_metrics["auprc"] > best_val_auc:
                best_val_auc = val_metrics["auprc"]
                torch.save(gnn_binn_model.state_dict(), best_model_path_fold)

        print(f"Repetition {repetition_idx + 1}, Fold {fold_idx + 1} GNN-BINN training complete. Best Val AUPRC: {best_val_auc:.4f}")
        gnn_binn_model.load_state_dict(torch.load(best_model_path_fold))

        _, test_metrics, _, _, y_true, y_pred_proba = validate_gnn_binn_epoch(gnn_binn_model, test_loader, criterion, device)

        print(f"GNN-BINN Test AUPRC: {test_metrics['auprc']:.4f}")
        wandb.log({f"fold_test/gnn_binn_{k}": v for k, v in test_metrics.items()})
        run.finish()


        return test_metrics, y_true, y_pred_proba



    def aggregate_fold_results(self, fold_metrics_list):
        df = pd.DataFrame(fold_metrics_list)
        mean_metrics = df.mean().to_dict()
        std_metrics = df.std().to_dict()
        summary = {}
        for key, value in mean_metrics.items(): summary[f"{key}_mean"] = value
        for key, value in std_metrics.items(): summary[f"{key}_std"] = value
        return summary


    def extract_from_loader(self, loader, num_nodes, input_features):
        print("Extracting and flattening data from loader...")
        X_list, y_list = [], []
        for batch in loader:
            batch_x = batch.x
            batch_y = batch.y
            batch_size = len(batch_y)
            reshaped_x = batch_x.numpy().reshape(batch_size, num_nodes, input_features)
            X_list.append(reshaped_x)
            y_list.append(batch_y.numpy().flatten())
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        X_flat = X.reshape(len(y), -1)

        print("Shape of flattened X:", X_flat.shape)
        print("Shape of y:", y.shape)

        return X_flat, y


    def print_comparison_results(self, all_summaries):
        print("\n" + "="*120)
        print("MODEL COMPARISON RESULTS (MEAN ± STD ACROSS FOLDS)")
        print("="*120)
        metrics = ['auprc', 'auc', 'accuracy', 'precision', 'recall', 'f1']
        print(f"{'Model':<20}" + " ".join([f"{m.upper():<18}" for m in metrics]))
        print("-" * 120)
        sorted_models = sorted(all_summaries.items(), key=lambda x: x[1]['auprc_mean'], reverse=True)
        for model_name, summary in sorted_models:
            row = f"{model_name:<20}"
            for m in metrics:
                mean = summary.get(f"{m}_mean", 0)
                std = summary.get(f"{m}_std", 0)
                row += f"{mean:.4f} ± {std:.4f}   "
            print(row)
        print("="*120)


    def perform_significance_tests(self, all_metrics, primary_metric='auprc'):
        """
        Performs paired t-tests between GNN-BINN and other models.
        """
        print("\n" + "="*80)
        print(f"STATISTICAL SIGNIFICANCE TESTING (PAIRED T-TEST) VS. GNN-BINN ON '{primary_metric.upper()}'")
        print("="*80)
        
        if 'GNN-BINN' not in all_metrics or not all_metrics['GNN-BINN']:
            print("GNN-BINN results not found. Skipping tests.")
            return

        gnn_binn_scores = [m[primary_metric] for m in all_metrics['GNN-BINN']]
        
        models_to_compare = ['BINN', 'FDNN', 'Logistic Regression', 'Decision Tree', 'Random Forest']
        num_comparisons = len(models_to_compare) 
        alpha = 0.05
        adjusted_alpha = alpha / num_comparisons
        
        for model_name in models_to_compare:
            if model_name not in all_metrics or not all_metrics[model_name]:
                continue
            
            model_scores = [m[primary_metric] for m in all_metrics[model_name]]
            
            if len(gnn_binn_scores) != len(model_scores):
                print(f"Skipping {model_name}: mismatched number of results.")
                continue

            # Perform the paired t-test
            t_statistic, p_value = ttest_rel(gnn_binn_scores, model_scores, alternative='greater')
            
            mean_gnn_binn = np.mean(gnn_binn_scores)
            mean_model = np.mean(model_scores)

            print(f"Comparing GNN-BINN (Mean: {mean_gnn_binn:.4f}) with {model_name} (Mean: {mean_model:.4f})")
            print(f"  T-statistic: {t_statistic:.8f}, P-value: {p_value:.8f}")

            if p_value < adjusted_alpha:
                print(f"  --> The difference is statistically significant (p < {adjusted_alpha:.8f}).")
            else:
                print(f"  --> The difference is NOT statistically significant (p >= {adjusted_alpha:.8f}).")
            print("-" * 80)


    def run_comparison(self):
        print("Starting Model Comparison Pipeline...")
        set_random_seed(self.random_seed)

        input_dim, hidden_dims, output_dim, binn_edge_index, data, features, pathway_net = self.prepare_initial_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("\n--- Preparing GNN-BINN Data Pipeline ---")
        X_gnn_full, y_gnn_full, edge_index, pos_weight_gnn_full, network_ordered_protein_ids = load_gnn_data(
            self.binn_config, self.gnn_binn_config, pathway_net, data, features)
        input_features = 2 if self.binn_config.data_type == 'combined' else 1
        num_nodes = len(network_ordered_protein_ids)
        
        print("\n--- Preparing Standard BINN & ML Data Pipeline ---")
        X_binn_full, y_binn_full, _, _ = load_binn_data(
            self.binn_config, pathway_net, data, features)

        all_metrics = {model: [] for model in self.plot_data.keys()}
        wandb_group_name = f"comparison_exp_{wandb.util.generate_id()}"
        
        for i in range(self.n_repeats):
            print(f"\n{'#'*25} Repetition {i + 1} / {self.n_repeats} {'#'*25}")
            
            # Set a new seed for each repetition to ensure different data splits
            repetition_seed = self.random_seed + i
            set_random_seed(repetition_seed)
            
            # Create new folds for the current repetition
            print(f"Creating new CV folds for repetition {i+1}...")
            kfold_gnn_dataloaders = create_gnn_cv_dataloaders(
                X_gnn_full, y_gnn_full, edge_index, num_nodes, input_features, config=self.gnn_binn_config, seed=repetition_seed)
            kfold_binn_dataloaders = create_binn_cv_dataloaders(
                X_binn_full, y_binn_full, config=self.binn_config, num_folds=self.gnn_binn_config.num_folds, seed=repetition_seed)

            zipped_loaders = zip(kfold_gnn_dataloaders, kfold_binn_dataloaders)

            for fold, ((gnn_train, gnn_val, gnn_test, pos_w_gnn), (binn_train, binn_val, binn_test, pos_w_binn)) in enumerate(zipped_loaders):
                print(f"\n{'='*20} Fold {fold + 1} / {self.gnn_binn_config.num_folds} {'='*20}")

                X_train_flat, y_train = self.extract_from_loader(gnn_train, num_nodes, input_features)
                X_val_flat, y_val = self.extract_from_loader(gnn_val, num_nodes, input_features)
                X_test_flat, y_test = self.extract_from_loader(gnn_test, num_nodes, input_features)

                # Train ML models
                dt_metrics, y_true_dt, y_pred_dt = self.train_ml_model('dt', X_train_flat, y_train, X_test_flat, y_test)
                all_metrics['Decision Tree'].append(dt_metrics)
                self.plot_data['Decision Tree'].append({'y_true': y_true_dt, 'y_pred': y_pred_dt})

                lr_metrics, y_true_lr, y_pred_lr = self.train_ml_model('lr', X_train_flat, y_train, X_test_flat, y_test)
                all_metrics['Logistic Regression'].append(lr_metrics)
                self.plot_data['Logistic Regression'].append({'y_true': y_true_lr, 'y_pred': y_pred_lr})

                rf_metrics, y_true_rf, y_pred_rf = self.train_ml_model('rf', X_train_flat, y_train, X_test_flat, y_test)
                all_metrics['Random Forest'].append(rf_metrics)
                self.plot_data['Random Forest'].append({'y_true': y_true_rf, 'y_pred': y_pred_rf})

                # Train FDNN
                fdnn_metrics, y_true_fdnn, y_pred_fdnn = self.train_fdnn_fold(
                    i, fold, X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test, 
                    pos_w_gnn, device, wandb_group_name
                )
                all_metrics['FDNN'].append(fdnn_metrics)
                self.plot_data['FDNN'].append({'y_true': y_true_fdnn, 'y_pred': y_pred_fdnn})

                # Train Standard BINN
                binn_metrics, y_true_binn, y_pred_binn = self.train_binn_fold(i, fold, binn_train, binn_val, binn_test, pos_w_binn, input_dim, hidden_dims, output_dim, binn_edge_index, device, wandb_group_name)
                all_metrics['BINN'].append(binn_metrics)
                self.plot_data['BINN'].append({'y_true': y_true_binn, 'y_pred': y_pred_binn})

                # Train GNN-BINN
                gnn_metrics, y_true_gnn, y_pred_gnn = self.train_gnn_binn_fold(i, fold, gnn_train, gnn_val, gnn_test, pos_w_gnn, input_dim, hidden_dims, output_dim, binn_edge_index, num_nodes, input_features, device, wandb_group_name)
                all_metrics['GNN-BINN'].append(gnn_metrics)
                self.plot_data['GNN-BINN'].append({'y_true': y_true_gnn, 'y_pred': y_pred_gnn})

        print("\n" + "="*50)
        print(f"AGGREGATING RESULTS FROM ALL {self.n_repeats} REPETITIONS AND {self.gnn_binn_config.num_folds} FOLDS")
        print("="*50)
        all_summaries = {model: self.aggregate_fold_results(metrics_list) for model, metrics_list in all_metrics.items() if metrics_list}
        
        
        
        self.print_comparison_results(all_summaries)

        self.perform_significance_tests(all_metrics, primary_metric='auprc')


        plot_comparison_curves(self, self.gnn_binn_config.plot_save_path)



def main():
    gnn_config = GNNConfig()
    binn_config = BinnConfig()
    gnn_binn_config = GNNBinnConfig()

    comparison = ModelComparison(binn_config, gnn_config, gnn_binn_config)
    comparison.run_comparison()

if __name__ == "__main__":
    main()