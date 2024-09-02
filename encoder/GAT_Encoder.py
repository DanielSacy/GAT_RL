import torch  # Import the PyTorch library for deep learning
import torch.nn as nn  # Import the PyTorch neural network module
from torch.nn import Linear, BatchNorm1d as BatchNorm  # Import the Linear and BatchNorm modules from PyTorch

# Custom GAT convolution layer is imported from a separate file named encoder.EdgeGATConv.py
from encoder.EdgeGATConv import EdgeGATConv


# Encoder that includes batch normalization and a residual connection around the custom GAT layer.
class ResidualEdgeGATEncoder(torch.nn.Module):
    """
    This class is a custom GAT encoder that includes batch normalization and a residual connection around the custom GAT layer.

    The input dimensions for nodes and edges are denoted as node_input_dim and edge_input_dim respectively. 
    The hidden dimension of the network, edge dimension used in the custom GAT convolution, number of layers, negative slope coefficient, and dropout rate are defined through their respective parameters.
    """

    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout):
        # Initialize the parent class (torch.nn.Module)
        super(ResidualEdgeGATEncoder, self).__init__()
        
        # Define the input dimensions and other parameters of the encoder
        self.node_input_dim = node_input_dim  # Input dimension for nodes
        self.edge_input_dim = edge_input_dim  # Input dimension for edges
        self.hidden_dim = hidden_dim  # Hidden dimension of the network
        self.edge_dim = edge_dim  # Edge dimension used in the custom GAT convolution
        self.layers = layers  # Number of layers in the encoder
        self.negative_slope = negative_slope  # Negative slope coefficient for the LeakyReLU activation function
        self.dropout = dropout  # Dropout rate for regularization

        # Initialize fully connected (dense) neural network layers
        self.fc_node = Linear(node_input_dim, hidden_dim)  # Fully connected layer for nodes
        self.fc_edge = Linear(edge_input_dim, edge_dim)  # Fully connected layer for edges
        
        # Add batch normalization to the node and edge features
        self.bn_node = BatchNorm(hidden_dim)  # Batch norm for nodes
        self.bn_edge = BatchNorm(edge_dim)  # Batch norm for edges

        # Define the custom GAT convolution layers
        self.edge_gat_layers = torch.nn.ModuleList(
            [EdgeGATConv(hidden_dim, hidden_dim, edge_dim, negative_slope, dropout) for _ in range(layers)]
        )
        
        # Initialize weights of all layers
        self.initialize_weights()

    def initialize_weights(self):
        """
        This function initializes the parameters of the encoder.
        
        Xavier initialization is used for weight matrices (typically dimension > 1), 
        and biases are initialized to zero. 
        Note that this only affects weights in linear layers, as specified by `fc_node` and `fc_edge`.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Typically applies to weight matrices
                nn.init.xavier_uniform_(param)  # Use Xavier initialization for weights
            elif 'bias' in name:  # Check if it's a bias term
                nn.init.constant_(param, 0)  # Initialize biases to zero
        torch.nn.init.constant_(self.fc_edge.bias, 0)
            
        
    def forward(self, data):
        """
        This function computes the node, edge, and graph embeddings.
        
        Parameters:
            data (object): The input data object containing node features (x), edge indices (edge_index), edge attributes (edge_attr), and batch information (batch).
            
        Returns:
            x (tensor): The computed node embeddings of shape (batch_size, num_nodes, hidden_dim)
        """
        # Extract the node features, edge indices, edge attributes, and batch information from the input data object
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Get the maximum batch size for later reshaping of the node embeddings
        batch_size = batch.max() + 1
        
        # Apply batch normalization and fully connected layer to the node features
        x = self.bn_node(self.fc_node(x))
        
        # Apply batch normalization and linear transformation to the edge attributes
        edge_attr = self.bn_edge(self.fc_edge(edge_attr))
                
        # Iterate through each Edge GAT layer, apply it with a residual connection, and accumulate the outputs
        for i, edge_gat_layer in enumerate(self.edge_gat_layers):
            x_next = edge_gat_layer(x, edge_index, edge_attr)
            x = x + x_next  # Accumulate the output of the current Edge GAT layer
            
        # Reshape the accumulated node embeddings to match the batch size
        x = x.reshape(batch_size, -1, self.hidden_dim) 
        return x  # Shape of x: (batch_size, num_nodes, hidden_dim)
