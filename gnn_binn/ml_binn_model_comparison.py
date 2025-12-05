import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as GNNDataLoader
from torch.utils.data import DataLoader as TorchDataLoader 

import wandb
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler



from config.binn_config import BinnConfig
from config.gnn_config import GNNConfig
from config.gnn_binn_config import GNNBinnConfig


from binn.src.model.create_neural_net import create_pathway_network, create_binn
from gnn_binn.src.string_files_preprocess import map_proteins_to_genes
from gnn_binn.src.model.gnn_binn import GNN_BINN


from gnn_binn.src.data_loader import (create_cv_dataloaders as create_gnn_cv_dataloaders,
                                      load_full_aligned_data as load_gnn_data)

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


def plot_comparison_curves(self, save_path=None):
        # generates and saves model comparison roc and auprc curves.
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"plots will be saved to: {os.path.abspath(save_path)}")

        sns.set_style("white")
        custom_colors = [
            '#457B9D',
            '#1A7431',
            '#FFF3B0',
            '#E09F3E',
            '#9E2A2B'
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
            ax_roc.plot(mean_fpr, mean_tpr, color=custom_colors[i],
                        label=f'{model_name}: {mean_auc:.2f} \u00B1 {std_auc:.2f}', lw=1.25)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color=custom_colors[i], alpha=0.1)

        ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.5)
        ax_roc.set(xlabel="1-specificity", ylabel="sensitivity")
        ax_roc.legend(loc="lower right", title="auc-roc", fontsize=13)
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
            ax_prc.plot(mean_recall, mean_precision, color=custom_colors[i],
                        label=f'{model_name}: {mean_auprc:.2f} \u00B1 {std_auprc:.2f}', lw=1.25)
            std_precision = np.std(precisions, axis=0)
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            ax_prc.fill_between(mean_recall, precision_lower, precision_upper, color=custom_colors[i], alpha=0.1)
        
        ax_prc.set(xlabel="recall", ylabel="precision")
        ax_prc.legend(loc="lower left", title="auprc", fontsize=13)
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
            'Decision Tree': [],
            'Logistic Regression': [],
            'Random Forest': []
        }

    def prepare_initial_data(self):
        print("preparing initial steps for data loading...")
        data, features, available_genes_in_data = load_initial_data(self.binn_config)
        available_genes_in_string = map_proteins_to_genes(
            self.gnn_config.protein_nodes_path, self.gnn_config.ppi_edges_path,
            self.gnn_config.mapping_file_path, self.gnn_config.gene_nodes_path,
            self.gnn_config.ggi_edges_path)
        
        input_dim, hidden_dims, output_dim, binn_edge_index, pathway_net = create_pathway_network(
            self.binn_config, available_genes_in_data, available_genes_in_string)
        
        return input_dim, hidden_dims, output_dim, binn_edge_index, data, features, pathway_net


    def train_ml_model(self, model_type, X_train, y_train, X_test, y_test):
        print(f"\ntraining {model_type.upper()}...")
        if model_type == 'dt':
            model = DecisionTreeClassifier(random_state=self.random_seed, max_depth=None, min_samples_split=20, min_samples_leaf=10, class_weight='balanced')
        elif model_type == 'lr':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = LogisticRegression(random_state=self.random_seed, max_iter=500, class_weight='balanced', C=1.0)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=90, random_state=self.random_seed, max_depth=None, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', n_jobs=-1)
        else:
            raise ValueError("invalid model_type")

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
        print(f"{model_type.upper()} auprc: {metrics['auprc']:.4f}")
        return metrics, y_test, y_pred_proba


    def train_binn_fold(self, repetition_idx, fold_idx, train_loader, val_loader, test_loader, pos_w, input_dim, hidden_dims, output_dim, binn_edge_index, device, wandb_group_name):
        # trains and evaluates the standard binn model for one fold of cross-validation.
        print(f"\n--- training binn for repetition {repetition_idx + 1}, fold {fold_idx + 1} ---")
        pos_w = pos_w.to(device)
        
        # setup wandb for the current fold
        run = setup_wandb_binn(self.binn_config, device, group=wandb_group_name, job_type=f'repeat_{repetition_idx+1}_binn_fold_{fold_idx+1}')
        
        # initialize model
        model = create_binn(input_dim, hidden_dims, output_dim, binn_edge_index).to(device)
        print(model)

        # initialize optimizer, scheduler, and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=self.binn_config.initial_lr, momentum=self.binn_config.momentum, weight_decay=self.binn_config.weight_decay, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.binn_config.num_epochs, eta_min=self.binn_config.min_lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_val_auc = 0.0
        best_epoch = 0
        best_model_path_fold = os.path.join(self.binn_config.model_save_path, f"{self.binn_config.model_name}_repeat_{repetition_idx+1}_fold_{fold_idx+1}.pth")

        # training loop
        for epoch in range(self.binn_config.num_epochs):
            print(f"epoch {epoch + 1}/{self.binn_config.num_epochs}")

            # training phase
            train_loss, train_metrics, _, _, _, _ = train_binn_epoch(model, train_loader, criterion, optimizer, device, epoch)

            # validation phase
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

            # save the best model based on validation auprc
            if val_metrics["auprc"] > best_val_auc:
                best_val_auc = val_metrics["auprc"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path_fold)

        print(f"repetition {repetition_idx + 1}, fold {fold_idx + 1} binn training complete. best epoch: {best_epoch} with val auprc: {best_val_auc:.4f}")

        # evaluate the best model on the test set
        model.load_state_dict(torch.load(best_model_path_fold))
        test_loss, test_metrics, _, _, y_true, y_pred_proba = validate_binn_epoch(
            model, test_loader, criterion, device
        )

        print("\n===== binn fold test set evaluation =====")
        print(f"test auprc: {test_metrics['auprc']:.4f}")

        # log test results and finish run
        wandb.log({f"fold_test/{k}": v for k, v in test_metrics.items()})
        run.finish()

        return test_metrics, y_true, y_pred_proba


    def train_gnn_binn_fold(self, repetition_idx, fold_idx, train_loader, val_loader, test_loader, pos_w, input_dim, hidden_dims, output_dim, binn_edge_index, num_nodes, input_features, device, wandb_group_name):
        print(f"\n--- training gnn-binn for repetition {repetition_idx + 1}, fold {fold_idx + 1} ---")
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
            print(f"epoch {epoch + 1}/{self.gnn_binn_config.num_epochs}")
            _, _, gnn_binn_model, _, _, _, _, _ = train_gnn_binn_epoch(gnn_binn_model, train_loader, criterion, optimizer, device, epoch)
            _, val_metrics, _, _, _, _ = validate_gnn_binn_epoch(gnn_binn_model, val_loader, criterion, device, epoch)
            scheduler.step()
            if val_metrics["auprc"] > best_val_auc:
                best_val_auc = val_metrics["auprc"]
                torch.save(gnn_binn_model.state_dict(), best_model_path_fold)

        print(f"repetition {repetition_idx + 1}, fold {fold_idx + 1} gnn-binn training complete. best val auprc: {best_val_auc:.4f}")
        gnn_binn_model.load_state_dict(torch.load(best_model_path_fold))

        _, test_metrics, _, _, y_true, y_pred_proba = validate_gnn_binn_epoch(gnn_binn_model, test_loader, criterion, device)

        print(f"gnn-binn test auprc: {test_metrics['auprc']:.4f}")
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
        print("extracting and flattening data from loader...")
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

        print("shape of flattened x:", X_flat.shape)
        print("shape of y:", y.shape)

        return X_flat, y


    def print_comparison_results(self, all_summaries):
        print("\n" + "="*120)
        print("model comparison results (mean ± std across folds)")
        print("="*120)
        metrics = ['auprc', 'auc', 'accuracy', 'precision', 'recall', 'f1']
        print(f"{'model':<20}" + " ".join([f"{m.upper():<18}" for m in metrics]))
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
        # performs paired t-tests between gnn-binn and other models.
        print("\n" + "="*80)
        print(f"statistical significance testing (paired t-test) vs. gnn-binn on '{primary_metric.upper()}'")
        print("="*80)
        
        if 'GNN-BINN' not in all_metrics or not all_metrics['GNN-BINN']:
            print("gnn-binn results not found. skipping tests.")
            return

        gnn_binn_scores = [m[primary_metric] for m in all_metrics['GNN-BINN']]
        
        models_to_compare = ['BINN', 'Logistic Regression', 'Decision Tree', 'Random Forest']
        num_comparisons = len(models_to_compare) 
        alpha = 0.05
        adjusted_alpha = alpha / num_comparisons
        
        for model_name in models_to_compare:
            if model_name not in all_metrics or not all_metrics[model_name]:
                continue
            
            model_scores = [m[primary_metric] for m in all_metrics[model_name]]
            
            if len(gnn_binn_scores) != len(model_scores):
                print(f"skipping {model_name}: mismatched number of results.")
                continue

            # perform the paired t-test
            t_statistic, p_value = ttest_rel(gnn_binn_scores, model_scores, alternative='greater')
            
            mean_gnn_binn = np.mean(gnn_binn_scores)
            mean_model = np.mean(model_scores)

            print(f"comparing gnn-binn (mean: {mean_gnn_binn:.4f}) with {model_name} (mean: {mean_model:.4f})")
            print(f"  t-statistic: {t_statistic:.8f}, p-value: {p_value:.8f}")

            if p_value < adjusted_alpha:
                print(f"  --> the difference is statistically significant (p < {adjusted_alpha:.8f}).")
            else:
                print(f"  --> the difference is not statistically significant (p >= {adjusted_alpha:.8f}).")
            print("-" * 80)


    def run_comparison(self):
        print("starting model comparison pipeline...")
        set_random_seed(self.random_seed)

        # shared initial data preparation 
        input_dim, hidden_dims, output_dim, binn_edge_index, data, features, pathway_net = self.prepare_initial_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device: {device}")

        # gnn-binn full data loading 
        print("\n--- preparing gnn-binn data pipeline ---")
        X_gnn_full, y_gnn_full, edge_index, pos_weight_gnn_full, network_ordered_protein_ids = load_gnn_data(
            self.binn_config, self.gnn_binn_config, pathway_net, data, features)
        input_features = 2 if self.binn_config.data_type == 'combined' else 1
        num_nodes = len(network_ordered_protein_ids)

        # standard binn & ml models full data loading
        print("\n--- preparing standard binn & ml data pipeline ---")
        X_binn_full, y_binn_full, _, _ = load_binn_data(
            self.binn_config, pathway_net, data, features)

        # repetitions and cross-validation loop
        all_metrics = {model: [] for model in self.plot_data.keys()}
        wandb_group_name = f"comparison_exp_{wandb.util.generate_id()}"
        
        for i in range(self.n_repeats):
            print(f"\n{'#'*25} repetition {i + 1} / {self.n_repeats} {'#'*25}")
            
            # set a new seed for each repetition to ensure different data splits
            repetition_seed = self.random_seed + i
            set_random_seed(repetition_seed)
            
            # create new folds for the current repetition
            print(f"creating new cv folds for repetition {i+1}...")
            kfold_gnn_dataloaders = create_gnn_cv_dataloaders(
                X_gnn_full, y_gnn_full, edge_index, num_nodes, input_features, config=self.gnn_binn_config, seed=repetition_seed)
            kfold_binn_dataloaders = create_binn_cv_dataloaders(
                X_binn_full, y_binn_full, config=self.binn_config, num_folds=self.gnn_binn_config.num_folds, seed=repetition_seed)

            zipped_loaders = zip(kfold_gnn_dataloaders, kfold_binn_dataloaders)

            for fold, ((gnn_train, gnn_val, gnn_test, pos_w_gnn), (binn_train, binn_val, binn_test, pos_w_binn)) in enumerate(zipped_loaders):
                print(f"\n{'='*20} fold {fold + 1} / {self.gnn_binn_config.num_folds} {'='*20}")

                X_train_flat, y_train = self.extract_from_loader(gnn_train, num_nodes, input_features)
                X_test_flat, y_test = self.extract_from_loader(gnn_test, num_nodes, input_features)

                # train ml models
                dt_metrics, y_true_dt, y_pred_dt = self.train_ml_model('dt', X_train_flat, y_train, X_test_flat, y_test)
                all_metrics['Decision Tree'].append(dt_metrics)
                self.plot_data['Decision Tree'].append({'y_true': y_true_dt, 'y_pred': y_pred_dt})

                lr_metrics, y_true_lr, y_pred_lr = self.train_ml_model('lr', X_train_flat, y_train, X_test_flat, y_test)
                all_metrics['Logistic Regression'].append(lr_metrics)
                self.plot_data['Logistic Regression'].append({'y_true': y_true_lr, 'y_pred': y_pred_lr})

                rf_metrics, y_true_rf, y_pred_rf = self.train_ml_model('rf', X_train_flat, y_train, X_test_flat, y_test)
                all_metrics['Random Forest'].append(rf_metrics)
                self.plot_data['Random Forest'].append({'y_true': y_true_rf, 'y_pred': y_pred_rf})

                # train standard binn
                binn_metrics, y_true_binn, y_pred_binn = self.train_binn_fold(i, fold, binn_train, binn_val, binn_test, pos_w_binn, input_dim, hidden_dims, output_dim, binn_edge_index, device, wandb_group_name)
                all_metrics['BINN'].append(binn_metrics)
                self.plot_data['BINN'].append({'y_true': y_true_binn, 'y_pred': y_pred_binn})

                # train gnn-binn
                gnn_metrics, y_true_gnn, y_pred_gnn = self.train_gnn_binn_fold(i, fold, gnn_train, gnn_val, gnn_test, pos_w_gnn, input_dim, hidden_dims, output_dim, binn_edge_index, num_nodes, input_features, device, wandb_group_name)
                all_metrics['GNN-BINN'].append(gnn_metrics)
                self.plot_data['GNN-BINN'].append({'y_true': y_true_gnn, 'y_pred': y_pred_gnn})

        # aggregate, print, and plot final results 
        print("\n" + "="*50)
        print(f"aggregating results from all {self.n_repeats} repetitions and {self.gnn_binn_config.num_folds} folds")
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