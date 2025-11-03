from dataclasses import dataclass, field

@dataclass
class GNNConfig:
    # STRING database files
    protein_nodes_path: str = r'data/string/protein_nodes.json'
    ppi_edges_path: str = r'data/string/ppi_edges.npy'
    mapping_file_path: str = r'data/string/filtered_mapping.txt'
    gene_nodes_path: str = r'data/string/gene_nodes.json'
    ggi_edges_path: str = r'data/string/ggi_edges.npy'

    filtered_ordered_nodes_path: str = r'data/string/filtered_ordered_gene_nodes.json'
    filtered_ordered_edges_path: str = r'data/string/filtered_ordered_ggi_edges.npy'

    # model
    hidden_dim: int = 64
    dropout_rate: float = 0.1

    # Dataset
    test_size: float = 0.1
    val_size: float = 0.1

    # training
    batch_size: int = 48
    num_epochs: int = 30
    initial_lr: float = 0.0002
    min_lr: float = 1e-5
    weight_decay: float = 0.001 # 0.0005
    momentum: float = 0.85

    # warmup
    warmup_epochs: int = 5

    optimizer: str = 'adam'  # the model was tested with this optimizer only
    scheduler: str = "CosineAnnealingLR" # the model was tested with this scheduler only

    loss_function: str = "bcelogits" # the model was tested with this loss function only

    # Early stopping
    patience: int = 20
    min_delta: float = 0.0001
    mode: str = 'max'
    restore_best_weights: bool = True

     # Wandb
    wandb_project: str = "gnn"

    # Saving the model
    model_save_path: str = "checkpoints/gnn"
    model_name: str = "gnn_model.pth"


    # reproducibility
    random_seed: int = 42

    # Hardware
    num_workers: int = 1