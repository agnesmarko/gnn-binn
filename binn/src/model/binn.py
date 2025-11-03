import torch
import torch.nn as nn
import torch.nn.functional as F


# tanh, layer norm, multi-head
class BINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 norm: str = 'layer', weight_init: str = 'custom'):
        super(BINN, self).__init__()

        # Dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]
        self.input_dim = input_dim 

        # Parameters
        self.norm = norm
        self.weight_init = weight_init

        self.weight = None
        self.bias = nn.Parameter(torch.zeros(self.total_nodes - self.input_dim))
        self.edge_index = None
        self.sparse_indices = None

        # Normalization
        if self.norm == 'layer':
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in hidden_dims
            ])
        elif self.norm == 'batch':
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims
            ])

        # Prediction heads for each hidden layer
        self.prediction_heads = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in hidden_dims
        ])

    def set_connections(self, edge_index):
        num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # Sparse weight matrix indices
        self.sparse_indices = edge_index.clone()

        # Weight initialization 
        if self.weight_init == 'custom':
            # Calculate fan_in for each node (how many inputs each node receives)
            unique_targets, counts = torch.unique(edge_index[1], return_counts=True)

            # Create a mapping from target node to fan_in
            node_to_fan_in = {}
            for node, count in zip(unique_targets.tolist(), counts.tolist()):
                node_to_fan_in[node] = count

            # Create a fan_in tensor matching edge_index order
            fan_in_per_edge = torch.tensor([node_to_fan_in.get(target.item(), 1) 
                                            for target in edge_index[1]])

            # Initialize weights based on fan_in
            # Xavier/Glorot initialization may work better with tanh
            weight_scales = 1.0 / torch.sqrt(fan_in_per_edge.float())
            self.weight = nn.Parameter(torch.randn(num_edges) * weight_scales)

        elif self.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        

    def forward(self, x, return_activations=False):
        batch_size = x.size(0)

        # Initialize activations
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x

        all_activations = [x]

        # Prepare sparse weights
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t()

        # For storing layer predictions
        layer_predictions = []

        prev_activations = activations.clone()

        # Process each layer
        for i in range(1, len(self.layer_dims)):
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # Calculate weighted sum
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # Apply bias
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # Apply activation for hidden layers
            if i < len(self.layer_dims) - 1:  # Hidden layer
                if self.norm == 'layer':
                    normalized = self.layer_norms[i-1](current_layer_weighted_sum + layer_bias)
                elif self.norm == 'batch':
                    normalized = self.batch_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)

                # Get prediction from this layer
                layer_pred = self.prediction_heads[i-1](activated_layer_output)
                layer_predictions.append(layer_pred)
            else:  # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # Update activations
            # activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            # prev_activations = activations.clone()
            activations = prev_activations.clone() # Start with a fresh copy
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            prev_activations = activations # This becomes the input for the next loop

            # Store the full activation tensor at each step
            if i < len(self.layer_dims) - 1: # only for hidden layers
                all_activations.append(activated_layer_output)

        # Average all layer predictions
        if layer_predictions:
            avg_prediction = torch.stack(layer_predictions).mean(dim=0)

            # Apply sigmoid to the averaged prediction  
            # - we are changing this, because the loss function is changed to BCEWithLogitsLoss
            # avg_prediction = torch.sigmoid(avg_prediction)

            if return_activations:
                return avg_prediction, all_activations
            return avg_prediction
        else:
            # Fallback to final layer output if no predictions (should never happen)
            output_layer_start = self.layer_indices[-2]
            output_layer_end = self.layer_indices[-1]
            if return_activations:
                return activations[:, output_layer_start:output_layer_end], all_activations
            return activations[:, output_layer_start:output_layer_end]


    def forward_from_layer(self, start_layer_idx, partial_activations, return_activations=False):
        batch_size = partial_activations.size(0)
        all_activations = [] if return_activations else None
        activations = partial_activations.clone()
        prev_activations = activations.clone()
        layer_predictions = []
        
        for i in range(start_layer_idx, len(self.layer_dims) - 1):  # Up to but not including output if starting mid
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]
            weighted_sum_all = torch.sparse.mm(self.sparse_weights.t(), prev_activations.t()).t()
            current_sum = weighted_sum_all[:, layer_start_idx:layer_end_idx]
            bias_start = layer_start_idx - self.input_dim
            bias_end = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start:bias_end]
            if i < len(self.layer_dims) - 2:  # Hidden (note: adjusted for start)
                if self.norm == 'layer':
                    normed = self.layer_norms[i-1](current_sum + layer_bias)
                elif self.norm == 'batch':
                    normed = self.batch_norms[i-1](current_sum + layer_bias)
                activated = torch.tanh(normed)
                layer_pred = self.prediction_heads[i-1](activated)
                layer_predictions.append(layer_pred)
                if return_activations:
                    all_activations.append(activated)
            else:  # Last hidden or output
                activated = current_sum + layer_bias
                if return_activations and i < len(self.layer_dims) - 2:
                    all_activations.append(activated)
            
            activations[:, layer_start_idx:layer_end_idx] = activated
            prev_activations = activations.clone()
        
        # Output layer if not reached
        if start_layer_idx < len(self.layer_dims) - 1:
            out_start = self.layer_indices[-2]
            out_end = self.layer_indices[-1]
            weighted_sum_all = torch.sparse.mm(self.sparse_weights.t(), prev_activations.t()).t()
            out_sum = weighted_sum_all[:, out_start:out_end]
            bias_start = out_start - self.input_dim
            out_bias = self.bias[bias_start:]
            output = out_sum + out_bias
            activations[:, out_start:out_end] = output
        
        if layer_predictions:
            avg_pred = torch.stack(layer_predictions).mean(dim=0)
            if return_activations:
                return avg_pred, all_activations
            return avg_pred
        else:
            output_slice = activations[:, self.layer_indices[-2]:]
            if return_activations:
                return output_slice, all_activations
            return output_slice


    def forward_from(self, start_layer_idx, activations, return_activations=False):
        # for the shap explainer, we need the list of intermediate activations 
        layer_predictions = []
        intermediate_acts = []
        
        prev_activations = activations.clone()

        for i in range(start_layer_idx + 1, len(self.layer_dims)):
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            sparse_weights_t = torch.sparse_coo_tensor(
                self.sparse_indices.to(activations.device),
                self.weight.to(activations.device),
                (self.total_nodes, self.total_nodes),
            ).t()

            weighted_sum = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_layer_weighted_sum = weighted_sum[:, layer_start_idx:layer_end_idx]
            
            bias_start = layer_start_idx - self.input_dim
            bias_end = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start:bias_end].to(activations.device)

            if i < len(self.layer_dims) - 1:  # Hidden layer
                if self.norm == 'layer':
                    normalized = self.layer_norms[i-1](current_layer_weighted_sum + layer_bias)
                elif self.norm == 'batch':
                    normalized = self.batch_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)
                
                layer_pred = self.prediction_heads[i-1](activated_layer_output)
                layer_predictions.append(layer_pred)
                intermediate_acts.append(activated_layer_output)
            else:  # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            prev_activations = activations.clone()

        avg_prediction = torch.stack(layer_predictions).mean(dim=0)

        if return_activations:
             return avg_prediction, intermediate_acts
        return avg_prediction
    

    def forward_for_shap(self, x):
        """
        A forward pass that matches the exact computation used during training/inference.
        Returns the averaged prediction from all hidden layers WITHOUT sigmoid.
        """
        batch_size = x.size(0)

        # Initialize activations
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x

        # Prepare sparse weights
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t()

        # For storing layer predictions
        layer_predictions = []
        prev_activations = activations.clone()

        # Process each layer
        for i in range(1, len(self.layer_dims)):
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # Calculate weighted sum
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # Apply bias
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # Apply activation for hidden layers
            if i < len(self.layer_dims) - 1:  # Hidden layer
                if self.norm == 'layer':
                    normalized = self.layer_norms[i-1](current_layer_weighted_sum + layer_bias)
                elif self.norm == 'batch':
                    normalized = self.batch_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)

                # Get prediction from this layer
                layer_pred = self.prediction_heads[i-1](activated_layer_output)
                layer_predictions.append(layer_pred)
            else:  # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # Update activations
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            prev_activations = activations.clone()

        # Average all layer predictions (matching your regular forward pass)
        if layer_predictions:
            avg_prediction = torch.stack(layer_predictions).mean(dim=0)
            return avg_prediction
        else:
            # Fallback to final layer output
            output_layer_start = self.layer_indices[-2]
            output_layer_end = self.layer_indices[-1]
            return activations[:, output_layer_start:output_layer_end]









class SparseNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SparseNetwork, self).__init__()

        # dimensions for all nodes in all layers
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        #  starting index for each layer
        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]

        #  edge connections
        self.edge_index = None
        self.weights = None
        self.bias = None

    def set_connections(self, edge_index):
        """
        edge_index should be a 2×E tensor defining all connections.
        """
        self.edge_index = edge_index

        # weights initialization for each edge
        num_edges = edge_index.size(1)
        self.weights = nn.Parameter(torch.Tensor(num_edges).normal_(0, 0.1))

        #  bias for each non-input node
        num_non_input_nodes = self.total_nodes - self.layer_indices[1]
        self.bias = nn.Parameter(torch.zeros(num_non_input_nodes))

    def forward(self, x):
        # activations tensor initialization
        activations = torch.zeros(self.total_nodes, device=x.device)

        #  input activations
        activations[:self.layer_indices[1]] = x

        # Process each layer
        for i in range(1, len(self.layer_indices)):
            layer_start = self.layer_indices[i]
            layer_end = self.layer_indices[i+1] if i+1 < len(self.layer_indices) else self.total_nodes

            # For each node in the current layer
            for node_idx in range(layer_start, layer_end):
                # edges that connect to this node
                edge_mask = self.edge_index[1] == node_idx
                source_indices = self.edge_index[0][edge_mask]
                edge_indices = torch.where(edge_mask)[0]

                if len(source_indices) > 0:
                    #  weights for these edges
                    edge_weights = self.weights[edge_indices]

                    #  activations from source nodes
                    source_activations = activations[source_indices]

                    weighted_sum = torch.sum(source_activations * edge_weights)

                    # add bias and apply activation
                    bias_idx = node_idx - self.layer_indices[1]
                    activations[node_idx] = F.relu(weighted_sum + self.bias[bias_idx])


        return activations[self.layer_indices[-2]:]

# tanh, layer norm, multi-head
class SparseNetworkMultiHeadLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SparseNetworkMultiHeadLayer, self).__init__()

        # dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]
        self.input_dim = input_dim 

        # Parameters
        self.weight = None
        self.bias = nn.Parameter(torch.zeros(self.total_nodes - self.input_dim))
        self.edge_index = None
        self.sparse_indices = None

        # Layer normalization instead of batch normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in hidden_dims
        ])

        # Prediction heads for each hidden layer
        self.prediction_heads = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in hidden_dims
        ])

    def set_connections(self, edge_index):
        num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # sparse weight matrix indices
        self.sparse_indices = edge_index.clone()

        # Better weight initialization 
        # Calculate fan_in for each node (how many inputs each node receives)
        unique_targets, counts = torch.unique(edge_index[1], return_counts=True)

        # Create a mapping from target node to fan_in
        node_to_fan_in = {}
        for node, count in zip(unique_targets.tolist(), counts.tolist()):
            node_to_fan_in[node] = count

        # Create a fan_in tensor matching edge_index order
        fan_in_per_edge = torch.tensor([node_to_fan_in.get(target.item(), 1) 
                                        for target in edge_index[1]])

        # Initialize weights based on fan_in
        # Xavier/Glorot initialization may work better with tanh
        weight_scales = 1.0 / torch.sqrt(fan_in_per_edge.float())
        self.weight = nn.Parameter(torch.randn(num_edges) * weight_scales)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize activations
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x

        # Prepare sparse weights
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t()

        # For storing layer predictions
        layer_predictions = []

        prev_activations = activations.clone()

        # Process each layer
        for i in range(1, len(self.layer_dims)):
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # Calculate weighted sum
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # Apply bias
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # Apply activation for hidden layers
            if i < len(self.layer_dims) - 1:  # Hidden layer
                normalized = self.layer_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)  # keep tanh activation


                # Get prediction from this layer
                layer_pred = self.prediction_heads[i-1](activated_layer_output)
                layer_predictions.append(layer_pred)
            else:  # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # Update activations
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            prev_activations = activations.clone()

        # Average all layer predictions
        if layer_predictions:
            avg_prediction = torch.stack(layer_predictions).mean(dim=0)

            # Apply sigmoid to the averaged prediction 
            avg_prediction = torch.sigmoid(avg_prediction)
            return avg_prediction
        else:
            # Fallback to final layer output if no predictions (should never happen)
            output_layer_start = self.layer_indices[-2]
            output_layer_end = self.layer_indices[-1]
            return activations[:, output_layer_start:output_layer_end]


# tanh, batch norm, multi-head
class SparseNetworkMultiHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SparseNetworkMultiHead, self).__init__()

        # dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]
        self.input_dim = input_dim 

        # Parameters
        self.weight = None
        self.bias = nn.Parameter(torch.zeros(self.total_nodes - self.input_dim))
        self.edge_index = None
        self.sparse_indices = None

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])

        # Prediction heads for each hidden layer
        self.prediction_heads = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in hidden_dims
        ])

    def set_connections(self, edge_index):
        num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # sparse weight matrix indices
        self.sparse_indices = edge_index.clone()

        # Better weight initialization 
        # Calculate fan_in for each node (how many inputs each node receives)
        unique_targets, counts = torch.unique(edge_index[1], return_counts=True)

        # Create a mapping from target node to fan_in
        node_to_fan_in = {}
        for node, count in zip(unique_targets.tolist(), counts.tolist()):
            node_to_fan_in[node] = count

        # Create a fan_in tensor matching edge_index order
        fan_in_per_edge = torch.tensor([node_to_fan_in.get(target.item(), 1) 
                                        for target in edge_index[1]])

        # Initialize weights based on fan_in
        # Xavier/Glorot initialization may work better with tanh
        weight_scales = 1.0 / torch.sqrt(fan_in_per_edge.float())
        self.weight = nn.Parameter(torch.randn(num_edges) * weight_scales)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize activations
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x

        # Prepare sparse weights
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t()

        # For storing layer predictions
        layer_predictions = []

        prev_activations = activations.clone()

        # Process each layer
        for i in range(1, len(self.layer_dims)):
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # Calculate weighted sum
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # Apply bias
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # Apply activation for hidden layers
            if i < len(self.layer_dims) - 1:  # Hidden layer
                normalized = self.batch_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)

                # Get prediction from this layer
                layer_pred = self.prediction_heads[i-1](activated_layer_output)
                layer_predictions.append(layer_pred)
            else:  # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # Update activations
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output
            prev_activations = activations.clone()

        # Average all layer predictions
        if layer_predictions:
            avg_prediction = torch.stack(layer_predictions).mean(dim=0)

            # Apply sigmoid to the averaged prediction 
            avg_prediction = torch.sigmoid(avg_prediction)
            return avg_prediction
        else:
            # Fallback to final layer output if no predictions (should never happen)
            output_layer_start = self.layer_indices[-2]
            output_layer_end = self.layer_indices[-1]
            return activations[:, output_layer_start:output_layer_end]


# version 3: tanh, batch norm
class SparseNetworkBatch(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SparseNetworkBatch, self).__init__()

        # dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]
        self.input_dim = input_dim 

        # Parameters (weight initialized in set_connections)
        self.weight = None
        self.bias = nn.Parameter(torch.zeros(self.total_nodes - self.input_dim)) 
        self.edge_index = None
        self.sparse_indices = None 

        # Replace layer normalization with batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])

    def set_connections(self, edge_index):
        num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # sparse weight matrix indices
        self.sparse_indices = edge_index.clone()

        # Better weight initialization 
        # Calculate fan_in for each node (how many inputs each node receives)
        unique_targets, counts = torch.unique(edge_index[1], return_counts=True)

        # Create a mapping from target node to fan_in
        node_to_fan_in = {}
        for node, count in zip(unique_targets.tolist(), counts.tolist()):
            node_to_fan_in[node] = count

        # Create a fan_in tensor matching edge_index order
        fan_in_per_edge = torch.tensor([node_to_fan_in.get(target.item(), 1) 
                                        for target in edge_index[1]])

        # Initialize weights based on fan_in
        # Xavier/Glorot initialization may work better with tanh
        weight_scales = 1.0 / torch.sqrt(fan_in_per_edge.float())
        self.weight = nn.Parameter(torch.randn(num_edges) * weight_scales)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize the main activations tensor that will be updated
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x # Initial input assignment

        # Prepare sparse weights 
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t() # Pre-transpose

        # Keep track of the activations from the *previous* step to use as input for matmul
        prev_activations = activations.clone() # initial activations

        for i in range(1, len(self.layer_dims)): # Loop from first hidden layer to output layer
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # Calculate weighted sum using 'prev_activations'
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()

            # Select the portion for the current layer
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # Bias and Activation 
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # Apply activation (tanh for hidden, none for output)
            if i < len(self.layer_dims) - 1: # Hidden layer
                # Apply batch norm and tanh activation
                normalized = self.batch_norms[i-1](current_layer_weighted_sum + layer_bias)
                activated_layer_output = torch.tanh(normalized)
            else: # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # Update activations tensor with current layer's output
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output

            # Prepare for next iteration
            prev_activations = activations.clone()

        # Return the output slice
        output_layer_start = self.layer_indices[-2]
        output_layer_end = self.layer_indices[-1]
        return activations[:, output_layer_start:output_layer_end]


# version 2: layer norm, relu
class SparseNetworkLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SparseNetworkLayer, self).__init__()

        #  dimensions
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        self.layer_indices = [0]
        for dim in self.layer_dims:
            self.layer_indices.append(self.layer_indices[-1] + dim)

        self.total_nodes = self.layer_indices[-1]
        self.input_dim = input_dim 

        # Parameters (weight initialized in set_connections)
        self.weight = None
        # Bias only for non-input nodes
        self.bias = nn.Parameter(torch.zeros(self.total_nodes - self.input_dim)) 
        self.edge_index = None
        self.sparse_indices = None 


        # Add layer normalization between layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in hidden_dims
        ])

    def set_connections(self, edge_index):
        num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # sparse weight matrix indices
        self.sparse_indices = edge_index.clone()

        # self.weight = nn.Parameter(torch.Tensor(num_edges).normal_(0, 0.1))

        
        # Better weight initialization 
        # Calculate fan_in for each node (how many inputs each node receives)
        # Count occurrences of each target node in edge_index[1]
        unique_targets, counts = torch.unique(edge_index[1], return_counts=True)

        # Create a mapping from target node to fan_in
        node_to_fan_in = {}
        for node, count in zip(unique_targets.tolist(), counts.tolist()):
            node_to_fan_in[node] = count

        # Create a fan_in tensor matching edge_index order
        fan_in_per_edge = torch.tensor([node_to_fan_in.get(target.item(), 1) 
                                        for target in edge_index[1]])

        # Initialize weights based on fan_in
        # Use Kaiming/He initialization principle adapted for sparse connections
        # This will scale weights differently for each connection based on target's fan_in
        weight_scales = 1.0 / torch.sqrt(fan_in_per_edge.float())
        self.weight = nn.Parameter(torch.randn(num_edges) * weight_scales)

        


    def forward(self, x):
        batch_size = x.size(0)

        # Initialize the main activations tensor that will be updated
        activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
        activations[:, :self.input_dim] = x # Initial input assignment

        # Prepare sparse weights 
        sparse_indices_dev = self.sparse_indices.to(x.device)
        weights_dev = self.weight.to(x.device)
        sparse_weights = torch.sparse_coo_tensor(
            sparse_indices_dev,
            weights_dev,
            (self.total_nodes, self.total_nodes),
            device=x.device
        )
        sparse_weights_t = sparse_weights.t() # Pre-transpose

        
        # Keep track of the activations from the *previous* step to use as input for matmul
        prev_activations = activations.clone() # initial activations

        for i in range(1, len(self.layer_dims)): # Loop from first hidden layer to output layer
            layer_start_idx = self.layer_indices[i]
            layer_end_idx = self.layer_indices[i+1]

            # Calculate weighted sum using 'prev_activations'
            # This uses the state *before* this layer's update was applied to 'activations'
            # The tensor prev_activations.t() is the one autograd needs to track correctly
            weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, prev_activations.t()).t()

            # Select the portion for the current layer
            current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

            # Bias and Activation 
            bias_start_index = layer_start_idx - self.input_dim
            bias_end_index = layer_end_idx - self.input_dim
            # Ensure bias is on the correct device
            layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

            # Apply activation (ReLU for hidden, none for output)
            if i < len(self.layer_dims) - 1: # Hidden layer
                activated_layer_output = F.relu(self.layer_norms[i-1](current_layer_weighted_sum + layer_bias))
            else: # Output layer
                activated_layer_output = current_layer_weighted_sum + layer_bias

            # Update the main 'activations' tensor INPLACE for the current layer 
            # This modifies 'activations'. Crucially, 'prev_activations' used above is NOT modified here.
            activations[:, layer_start_idx:layer_end_idx] = activated_layer_output

            # Prepare 'prev_activations' for the NEXT iteration 
            # After 'activations' has been updated with this layer's result,
            # clone it so the *next* iteration's sparse_mm uses the correct, updated input state.
            prev_activations = activations.clone()


        # Return the output slice from the final 'activations' state
        output_layer_start = self.layer_indices[-2]
        output_layer_end = self.layer_indices[-1]
        return activations[:, output_layer_start:output_layer_end]


    # def forward(self, x):
    #     batch_size = x.size(0)

    #     if self.weight is None or self.sparse_indices is None:
    #         raise RuntimeError("Network connections not set. Call set_connections(edge_index) first.")

    #     # --- Use 'current_activations' to hold the state for the current step ---
    #     current_activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)
    #     current_activations[:, :self.input_dim] = x

    #     sparse_indices_dev = self.sparse_indices.to(x.device)
    #     weights_dev = self.weight.to(x.device)
    #     sparse_weights = torch.sparse_coo_tensor(
    #         sparse_indices_dev,
    #         weights_dev,
    #         (self.total_nodes, self.total_nodes),
    #         device=x.device
    #     )
    #     # Pre-transpose the sparse matrix if used repeatedly
    #     sparse_weights_t = sparse_weights.t()

    #     # --- Process layers ---
    #     # Create a tensor to build the next state without modifying the one used for calculation
    #     next_activations = current_activations.clone() # Start with a copy

    #     for i in range(1, len(self.layer_dims)):
    #         layer_start_idx = self.layer_indices[i]
    #         layer_end_idx = self.layer_indices[i+1]

    #         # --- Calculate weighted sum using 'current_activations' ---
    #         # Ensure the tensor used here (current_activations) is not modified inplace below
    #         weighted_sum_all_nodes = torch.sparse.mm(sparse_weights_t, current_activations.t()).t()

    #         current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

    #         bias_start_index = layer_start_idx - self.input_dim
    #         bias_end_index = layer_end_idx - self.input_dim
    #         layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device)

    #         if i < len(self.layer_dims) - 1: # Hidden layer
    #             activated_layer_output = F.relu(current_layer_weighted_sum + layer_bias)
    #         else: # Output layer
    #             activated_layer_output = current_layer_weighted_sum + layer_bias

    #         # --- Update the 'next_activations' tensor, NOT 'current_activations' ---
    #         next_activations[:, layer_start_idx:layer_end_idx] = activated_layer_output

    #         # --- Prepare for the next iteration ---
    #         # Update current_activations ONLY AFTER its use in this iteration is complete
    #         # Cloning ensures that the tensor used in the *next* iteration's sparse_mm
    #         # is distinct from the one used in the *previous* iteration's sparse_mm,
    #         # preventing the inplace error.
    #         current_activations = next_activations.clone()


    #     # Return the output slice from the final state
    #     output_layer_start = self.layer_indices[-2]
    #     output_layer_end = self.layer_indices[-1]
    #     # Use the final 'current_activations' which holds the complete result
    #     return current_activations[:, output_layer_start:output_layer_end]

    # def forward(self, x):
    #     # x shape: [batch_size, input_dim]
    #     batch_size = x.size(0)

    #     # Ensure parameters are initialized
    #     if self.weight is None or self.sparse_indices is None:
    #         raise RuntimeError("Network connections not set. Call set_connections(edge_index) first.")

        
    #     # Shape: [batch_size, total_nodes]
    #     activations = torch.zeros(batch_size, self.total_nodes, device=x.device, dtype=x.dtype)

    #     # Assign input features for the whole batch to the first 'input_dim' columns
    #     activations[:, :self.input_dim] = x 

        
    #     # Ensure indices and values are on the correct device
    #     sparse_indices_dev = self.sparse_indices.to(x.device)
    #     weights_dev = self.weight.to(x.device)

    #     # sparse_weights[i, j] = weight means connection FROM node j TO node i
    #     sparse_weights = torch.sparse_coo_tensor(
    #         sparse_indices_dev, 
    #         weights_dev, 
    #         (self.total_nodes, self.total_nodes),
    #         device=x.device
    #     )

        
    #     # Iterate through layers where computations happen (hidden + output)
    #     # Layer indices correspond to the END of the layer in the flattened structure
    #     for i in range(1, len(self.layer_dims)): # Loop from first hidden layer to output layer
    #         layer_start_idx = self.layer_indices[i]
    #         layer_end_idx = self.layer_indices[i+1]

    #         # Calculate weighted sum for ALL nodes using matrix multiplication
    #         # We need input = activations @ W^T for batch processing
    #         # Shape: [B, N] @ [N, N] -> [B, N] 
    #         # This computes the input signal arriving at each node
    #         weighted_sum_all_nodes = torch.sparse.mm(sparse_weights.t(), activations.t()).t()
    #         # Alternative using dense matmul if sparse is slow or causing issues:
    #         # weighted_sum_all_nodes = activations @ sparse_weights.to_dense().t() 

    #         # Select the weighted sums corresponding ONLY to the nodes in the current layer
    #         current_layer_weighted_sum = weighted_sum_all_nodes[:, layer_start_idx:layer_end_idx]

    #         # --- Bias and Activation for the current layer ---
    #         # Bias indices relative to the start of the bias vector (which excludes input nodes)
    #         bias_start_index = layer_start_idx - self.input_dim
    #         bias_end_index = layer_end_idx - self.input_dim

    #         # Select the relevant bias terms
    #         layer_bias = self.bias[bias_start_index:bias_end_index].to(x.device) # Shape [layer_dim]

    #         # Apply activation function (ReLU for hidden layers)
    #         # Note: For the *output* layer in binary classification, you typically DON'T apply ReLU.
    #         # The nn.BCEWithLogitsLoss expects raw logits.

    #         if i < len(self.layer_dims) - 1: # If it's a hidden layer
    #             activated_layer_output = F.relu(current_layer_weighted_sum + layer_bias)
    #         else: # If it's the output layer
    #             activated_layer_output = current_layer_weighted_sum + layer_bias # Output raw logits

    #         # Update the activations tensor for the current layer across the batch
    #         activations[:, layer_start_idx:layer_end_idx] = activated_layer_output

    #     # Return the activations of the output layer for the whole batch
    #     output_layer_start = self.layer_indices[-2]
    #     output_layer_end = self.layer_indices[-1] 
    #     return activations[:, output_layer_start:output_layer_end] # Shape: [batch_size, output_dim]

    # def forward(self, x):
    #     # x shape: [batch_size, input_dim]
    #     batch_size = x.size(0)

    #     # activations initialization with inputs
    #     activations = torch.zeros(self.total_nodes, device=x.device)
    #     activations[:self.layer_indices[1]] = x

    #     #  sparse weight matrix
    #     values = self.weight
    #     indices = self.sparse_indices
    #     sparse_weights = torch.sparse_coo_tensor(
    #         indices, values, (self.total_nodes, self.total_nodes)
    #     )

    #     # process each layer
    #     for i in range(1, len(self.layer_indices)):
    #         layer_start = self.layer_indices[i]
    #         layer_end = self.layer_indices[i+1] if i+1 < len(self.layer_indices) else self.total_nodes

    #         # sparse matrix multiplication for this layer
    #         weighted_sum = torch.sparse.mm(sparse_weights, activations.unsqueeze(1)).squeeze(1)

    #         #  bias and activation
    #         bias_indices = torch.arange(layer_start, layer_end) - self.layer_indices[1]
    #         activations[layer_start:layer_end] = F.relu(
    #             weighted_sum[layer_start:layer_end] + self.bias[bias_indices]
    #         )

    #     return activations[self.layer_indices[-2]:]
