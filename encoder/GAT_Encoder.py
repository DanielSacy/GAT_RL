import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d as BatchNorm
from src_batch.encoder.EdgeGATConv import EdgeGATConv


# Encoder that includes batch normalization and a residual connection around the custom GAT layer.
class ResidualEdgeGATEncoder(torch.nn.Module):
    """
    This class is a custom GAT encoder that includes batch normalization and a residual connection around the custom GAT layer."""
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout):
        super(ResidualEdgeGATEncoder, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.layers = layers
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.fc_node = Linear(node_input_dim, hidden_dim)
        self.fc_edge = Linear(edge_input_dim, edge_dim)
        self.bn_node = BatchNorm(hidden_dim)
        self.bn_edge = BatchNorm(edge_dim)

        self.edge_gat_layers = torch.nn.ModuleList(
            [EdgeGATConv(hidden_dim, hidden_dim, edge_dim, negative_slope, dropout) for _ in range(layers)]
        )
        
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
        torch.nn.init.constant_(self.fc_edge.bias, 0)
            
        
    def forward(self, data):
        """This function computes the node, edge, and graph embeddings."""
        x, edge_index, edge_attr, demand = data.x, data.edge_index, data.edge_attr, data.demand
        batch_size = data.num_graphs
        
        # Adding demand as a node feature
        x = torch.cat([x, demand], dim=-1)
        
        # Node and edge embedding
        x = self.bn_node(self.fc_node(x))
        edge_attr = self.bn_edge(self.fc_edge(edge_attr))
                
        # Apply Edge GAT with residual connection
        for edge_gat_layer in self.edge_gat_layers:
            x_next = edge_gat_layer(x, edge_index, edge_attr)
            x = x + x_next
        x = x.reshape(batch_size, -1, self.hidden_dim) 
        return x # Shape of x: (batch_size, num_nodes, hidden_dim)