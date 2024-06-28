import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d as BatchNorm, Sequential, ReLU
import torch_geometric
from src_batch.encoder.EdgeGATConv import EdgeGATConv


# Encoder that includes batch normalization and a residual connection around the custom GAT layer.
class ResidualEdgeGATEncoder(torch.nn.Module):
    """
    This class is a custom GAT encoder that includes batch normalization and a residual connection around the custom GAT layer."""
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads):
        super(ResidualEdgeGATEncoder, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.layers = layers
        
        self.fc_node = Linear(node_input_dim, hidden_dim * heads)
        self.fc_edge = Linear(edge_input_dim, hidden_dim * heads)
        self.fc_out = Linear(hidden_dim * heads, hidden_dim)
        self.bn_node = BatchNorm(hidden_dim * heads)
        self.bn_edge = BatchNorm(hidden_dim * heads)

        self.edge_gat_convs = torch.nn.ModuleList(
            [EdgeGATConv(hidden_dim, hidden_dim, hidden_dim, dropout=dropout, heads=heads) for _ in range(layers)]
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        This function initializes the parameters of the encoder.
        """
        torch.nn.init.xavier_uniform_(self.fc_node.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_edge.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_out.weight.data)
            
        
    def forward(self, data):
        """This function computes the node, edge, and graph embeddings."""
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node and edge embedding
        x = self.bn_node(self.fc_node(x))
        edge_attr = self.bn_edge(self.fc_edge(edge_attr))
        
        # Test later if applying a relu here is going to improve the performance
        # x = F.relu(self.bn_node(self.fc_node(x)))
        # edge_attr = F.relu(self.bn_edge(self.fc_edge(edge_attr)))
        
        # Apply Edge GAT convolution with residual connection
        for i, edge_gat_conv in enumerate(self.edge_gat_convs):
            x_residual = x
            x = edge_gat_conv(x, edge_index, edge_attr)
            # x = F.elu(x) # Test later if this is necessary
            x = x + x_residual
        
        # Apply a linear layer to the output to return it to the original dimension        
        x = self.fc_out(x)
        
        # Compute graph-level embedding by averaging all node embeddings
        # graph_embedding = torch_geometric.nn.global_mean_pool(x, batch)

        return x #, edge_attr, graph_embedding