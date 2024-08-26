import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F

# Custom GAT layer that includes edge features in the computation of attention coefficients.
class EdgeGATConv(MessagePassing):
    """
    This class is a custom GAT layer that includes edge features in the computation of attention coefficients.
    It also uses multi-head attention to improve the performance of the model.
    """
    def __init__(self, node_channels, hidden_dim, edge_dim, negative_slope, dropout, concat=True):
        super(EdgeGATConv, self).__init__(aggr='add' if concat else 'mean') 
        self.node_channels = node_channels
        self.hidden_dim = hidden_dim 
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat
        
        self.fc = torch.nn.Linear(node_channels, hidden_dim)
        self.att_vector = torch.nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim)
        # self.att_vector = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        """
        This function initializes the parameters of the encoder.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Typically applies to weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # Check if it's a bias term
                nn.init.constant_(param, 0)  # Initialize biases to zero
        
    def forward(self, x, edge_index, edge_attr, size=None):
        """This function computes the node embeddings."""
        x = self.fc(x)
                
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
    
    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        """This function computes the attention coefficients and returns the message to be aggregated."""
        # Concatenate features
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute attention coefficients
        alpha = self.att_vector(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        
        # Sample attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) 
        
        return x_j * alpha

    def update(self, aggr_out):
        """This function updates the node embeddings."""
        return aggr_out