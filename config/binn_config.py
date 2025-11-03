from dataclasses import dataclass, field

@dataclass
class BinnConfig:
    # File paths for creating the model 
    obo_file_path = r'data/ProstateCancer/go-basic.obo'
    gaf_file_path = r'data/ProstateCancer/goa_human.gaf'
    protein_input_nodes_path = r'data/ProstateCancer/cnv_input_genes.csv'
    preprocessed_gaf_file_path = r'data/ProstateCancer/goa_human_processed_filtered_8.gaf'

    max_level: int = 8  # max level of the GO hierarchy to include

    # Dataset
    features_path = r'data/ProstateCancer/pnet_index.csv'
    features_names: list = field(default_factory=lambda: ["gene", "type"])
    x_path = r'data/ProstateCancer/pnet_x.npy'
    y_path = r'data/ProstateCancer/pnet_y.npy'

    val_size: float = 0.1
    test_size: float = 0.1

    # data type to include in the training
    data_type: str = 'cnv'  # 'cnv', 'mut', or 'combined'


    # Model
    norm: str = "layer" # 'batch' or 'layer'
    weight_init: str = "custom"
    output_dim: int = 1  # binary classification

    # the list can include 'biological_process', 'molecular_function', 'cellular_component'
    root_nodes_to_include: list = field(default_factory=lambda: ['biological_process', 'molecular_function'])


    # Training
    batch_size: int = 64
    initial_lr: float = 1e-2
    min_lr: float = 1e-5
    weight_decay: float = 0.001
    num_epochs: int = 40
    momentum: float = 0.85

    optimizer: str = 'sgd'  # the model was tested with this optimizer only
    scheduler: str = "CosineAnnealingLR" # the model was tested with this scheduler only

    loss_function: str = "bcelogitsloss" # the model was tested with this loss function only

    # Early stopping
    patience: int = 20
    min_delta: float = 0.0001
    mode: str = 'max'
    restore_best_weights: bool = True

    # Reproducibility
    random_seed: int = 42


    # Wandb
    wandb_project: str = "binn_cv"
    log_every_n_batches: int | None = None  # if None, logging only happens at the end of each epoch

    # Saving the model
    model_save_path: str = "checkpoints/binn"
    model_name: str = "binn_model.pth"

    # Pretrained model path (if any)
    pretrained_model_path: str | None = None #r'checkpoints/binn/binn_model_139.pth'

    # Data split saving
    load_data_splits: bool = False
    save_data_splits: bool = False
    data_split_save_path: str = r'data/ProstateCancer/data_splits.npz'

    # Dense model comparison
    k_fold: int = 5
    dense_initial_lr: float = 0.00002

    # k-fold cross-validation
    num_folds: int = 10

    # Hardware
    num_workers: int = 1