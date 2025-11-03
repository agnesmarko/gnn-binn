import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
import torch_geometric

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from binn.src.model.binn import BINN


class GNNFeatureExtractor(torch.nn.Module):
    def __init__(self, input_features: int, hidden_dim: int = 64, output_features: int = 2, dropout_rate: float = 0.2,
                 aggr_method: str = "mean"):
        super(GNNFeatureExtractor, self).__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # GNN Layers
        self.conv1 = SAGEConv(input_features, hidden_dim, aggr=aggr_method)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr=aggr_method)
        self.conv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr=aggr_method)


        # Batch normalization
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim * 2)
        self.bn3 = BatchNorm(hidden_dim * 4)

        # learned projection head
        self.projection_head = nn.Linear(hidden_dim * 4, output_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)


        x = self.projection_head(x)

        return x


class GNN_BINN(torch.nn.Module):
    def __init__(self, gnn_input_features: int, gnn_hidden_dim: int, gnn_output_features: int, binn_model: BINN, num_nodes: int,
                 dropout_rate: float = 0.2, aggr_method: str = "mean"):
        super(GNN_BINN, self).__init__()
        
        self.num_nodes = num_nodes
        
        # GNN part for feature enrichment
        self.gnn_feature_extractor = GNNFeatureExtractor(
            input_features=gnn_input_features,
            hidden_dim=gnn_hidden_dim,
            output_features=gnn_output_features,
            dropout_rate=dropout_rate,
            aggr_method=aggr_method
        )
        
        # The output dimension of the GNN's feature extractor
        self.gnn_output_features = gnn_output_features
        
        # BINN part for classification
        self.binn = binn_model

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass data through the GNN to get enriched node features
        enriched_features = self.gnn_feature_extractor(x, edge_index)
        
        # Enriched features will have shape [num_nodes_in_batch, gnn_output_features]
        
        # We need to reshape these features for the BINN's input layer.
        # The input to the BINN is expected to be of shape [batch_size, num_nodes * num_features]
    
        
        # get the batch size
        batch_size = batch.max().item() + 1
        
        # Reshape the enriched features into [batch_size, num_nodes, gnn_output_features]
        # This assumes the nodes for each graph are contiguous in the batch
        reshaped_features = enriched_features.view(batch_size, self.num_nodes, self.gnn_output_features)
        
        # We need to flatten it in the specific order the BINN expects:
        # [cnv_gene1, cnv_gene2, ..., mut_gene1, mut_gene2, ...]
        # becomes
        # [enriched_feat1_gene1, enriched_feat1_gene2, ..., enriched_feat2_gene1, ... ]
        
        # permute the dimensions to [batch_size, gnn_output_features, num_nodes]
        permuted_features = reshaped_features.permute(0, 2, 1)
        
        # flatten it to [batch_size, gnn_output_features * num_nodes]
        binn_input = permuted_features.contiguous().view(batch_size, -1)
        
        # Pass the reshaped features to the BINN
        output = self.binn(binn_input)
        
        return output