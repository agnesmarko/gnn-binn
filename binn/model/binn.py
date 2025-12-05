import torch
import torch.nn as nn
import torch.nn.functional as F


class BINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 norm: str = 'layer', weight_init: str = 'custom'):
        super(BINN, self).__init__()

        # dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]
        self.input_dim = input_dim 

        # parameters
        self.norm = norm
        self.weight_init = weight_init

        self.weight = None
        self.bias = nn.Parameter(torch.zeros(self.total_nodes - self.input_dim))
        self.edge_index = None
        self.sparse_indices = None

        # normalization
        if self.norm == 'layer':
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in hidden_dims
            ])
        elif self.norm == 'batch':
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims
            ])

        # prediction heads for each hidden layer
        self.prediction_heads = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in hidden_dims
        ])

    def set_connections(self, edge_index):
        num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # sparse weight matrix indices
        self.sparse_indices = edge_index.clone()

        # weight initialization 
        if self.weight_init == 'custom':
            # calculate fan_in for each node (how many inputs each node receives)
            unique_targets, counts = torch.unique(edge_index[1], return_counts=True)

            # create a mapping from target node to fan_in
            node_to_fan_in = {}
            for node, count in zip(unique_targets.tolist(), counts.tolist()):
                node_to_fan_in[node] = count

            # create a fan_in tensor matching edge_index order
            fan_in_per_edge = torch.tensor([node_to_fan_in.get(target.item(), 1) 
                                            for target in edge_index[1]])

            # initialize weights based on fan_in
            # Xavier/Glorot initialization may work better with tanh
            weight_scales = 1.0 / torch.sqrt(fan_in_per_edge.float())
            self.weight = nn.Parameter(torch.randn(num_edges) * weight_scales)

        elif self.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        

    def forward(self, x, return_activations=False):
        batch_size = x.size(0)

        # initialize activations
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x

        all_activations = [x]

        # prepare sparse weights
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t()

        # for storing layer predictions
        layer_predictions = []

        prev_activations = activations.clone()

        # process each layer
        for i in range(1, len(self.layer_dims)):
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # calculate weighted sum
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # apply bias
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # apply activation for hidden layers
            if i < len(self.layer_dims) - 1:  # hidden layer
                if self.norm == 'layer':
                    normalized = self.layer_norms[i-1](current_layer_weighted_sum + layer_bias)
                elif self.norm == 'batch':
                    normalized = self.batch_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)

                # get prediction from this layer
                layer_pred = self.prediction_heads[i-1](activated_layer_output)
                layer_predictions.append(layer_pred)
            else:  # output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # update activations
            activations = prev_activations.clone() # a fresh copy
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            prev_activations = activations # this becomes the input for the next loop

            # store the full activation tensor at each step
            if i < len(self.layer_dims) - 1: # only for hidden layers
                all_activations.append(activated_layer_output)

        # average all layer predictions
        if layer_predictions:
            # stack predictions for averaging and for returning
            stacked_predictions = torch.stack(layer_predictions) # shape: [num_hidden_layers, batch_size, output_dim]
            avg_prediction = stacked_predictions.mean(dim=0)

            # squeeze the last dimension if output_dim is 1
            if stacked_predictions.size(-1) == 1:
                stacked_predictions = stacked_predictions.squeeze(-1)

            # return both the average and the individual predictions
            # the returned stacked_predictions will have shape: [num_hidden_layers, batch_size]
            if return_activations:
                return avg_prediction, stacked_predictions, all_activations
            return avg_prediction, stacked_predictions
        else:
            # fallback to final layer output if no predictions (should never happen)
            output_layer_start = self.layer_indices[-2]
            output_layer_end = self.layer_indices[-1]
            if return_activations:
                return activations[:, output_layer_start:output_layer_end], all_activations
            return activations[:, output_layer_start:output_layer_end]


