from dataclasses import dataclass, field

@dataclass

class GNNBinnConfig:
    # STRING database files
    protein_nodes_path: str = r'data/string/protein_nodes.json'
    ppi_edges_path: str = r'data/string/ppi_edges.npy'
    mapping_file_path: str = r'data/string/filtered_mapping.txt'
    gene_nodes_path: str = r'data/string/gene_nodes.json'
    ggi_edges_path: str = r'data/string/ggi_edges.npy'

    filtered_ordered_nodes_path: str = r'data/string/filtered_ordered_gene_nodes.json'
    filtered_ordered_edges_path: str = r'data/string/filtered_ordered_ggi_edges.npy'

    # gnn model
    hidden_dim: int = 8
    dropout_rate: float = 0.1
    aggregation_method: str = "mean"  # other options: "mean", "sum"

    # Dataset
    test_size: float = 0.1
    val_size: float = 0.1

    # training
    batch_size: int = 64
    initial_lr: float = 0.0006 # 2e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.00
    num_epochs: int = 60
    momentum: float = 0.85

    

    optimizer: str = 'adam'  # the model was tested with this optimizer only
    scheduler: str = "CosineAnnealingLR" # the model was tested with this scheduler only

    loss_function: str = "bcelogitsloss" # the model was tested with this loss function only

    # Early stopping
    patience: int = 10
    min_delta: float = 0.0001
    mode: str = 'max'
    restore_best_weights: bool = True

    # warmup
    warmup_epochs: int = 5

     # Wandb
    wandb_project: str = "gnn_binn_cv"

    # Saving the model
    model_save_path: str = "checkpoints/gnn_binn"
    model_name: str = "gnn_binn_model.pth"

    # Pretrained model path (if any)
    pretrained_model_path: str | None = "checkpoints/gnn_binn/gnn_binn_model_60.pth"



    # reproducibility
    random_seed: int = 42

    # k-fold cross-validation
    num_folds: int = 10
    plot_save_path: str = "data/plots"

    # Hardware
    num_workers: int = 1