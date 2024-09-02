# Import necessary PyTorch modules and custom model classes
import torch
from torch import nn

# Import custom encoder and decoder models from other files
from encoder.GAT_Encoder import ResidualEdgeGATEncoder  # Custom GNN encoder with residual connections and gated attention mechanism
from decoder.GAT_Decoder import GAT_Decoder  # Custom GAT-based decoder for graph embeddings

# Define the main model class that inherits from PyTorch's nn.Module
class Model(nn.Module):
    """
    Main model class that combines the encoder and decoder components.
    
    Args:
        node_input_dim (int): Input dimension of node features.
        edge_input_dim (int): Input dimension of edge features.
        hidden_dim (int): Hidden dimension for the GNN encoder and decoder.
        edge_dim (int): Dimension of edge embeddings.
        layers (int): Number of layers in the GNN encoder.
        negative_slope (float): Negative slope coefficient for LeakyReLU activation function.
        dropout (float): Dropout rate for the GNN encoder.
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout):
        # Call the parent class's constructor to initialize the model
        super(Model, self).__init__()
        
        # Initialize the custom encoder component with given parameters
        self.encoder = ResidualEdgeGATEncoder(node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout)
        
        # Initialize the custom decoder component with given parameters
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)

    def forward(self, data, n_steps, greedy, T):
        """
        Forward pass method that computes graph embeddings and generates actions.
        
        Args:
            data (torch_geometric.data.Data): Input data object containing node features, edge features, and edge indices.
            n_steps (int): Number of steps for the decoder to generate actions.
            greedy (bool): Whether to use greedy or sampling-based action selection.
            T (float): Temperature parameter for entropy regularization. Temperature for softmax based on Kun et al. (2021)
        
        Returns:
            actions (torch.Tensor): Generated actions based on graph embeddings.
            log_p (torch.Tensor): Log probability of generated actions.
        """

        # Compute node embeddings using the encoder component
        # data.x: node features, data.edge_attr: edge features, data.edge_index: edge indices
        x = self.encoder(data)  # Shape: (n_nodes, hidden_dim)
        
        # Compute graph embedding by taking the mean of all node embeddings per feature dimension
        graph_embedding = x.mean(dim=1)  # Shape: (batch_size, hidden_dim)

        # Get demand and capacity from input data object
        batch_size = data.batch.max().item() + 1
        demand = data.demand.reshape(batch_size, -1).float().to(data.x.device)
        capacity = data.capacity.reshape(batch_size, -1).float().to(data.x.device)
        
        # Call the decoder component to generate actions and compute log probability
        actions, log_p = self.decoder(x, graph_embedding, capacity, demand, n_steps, T, greedy)
        
        return actions, log_p