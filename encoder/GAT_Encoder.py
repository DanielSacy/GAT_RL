import torch
from typing import Any, List, Tuple
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d as BatchNorm, Sequential, ReLU
import torch_geometric
from src_batch.encoder.EdgeGATConv import EdgeGATConv


# Encoder that includes batch normalization and a residual connection around the custom GAT layer.
class ResidualEdgeGATEncoder(torch.nn.Module):
    """
    This class is a custom GAT encoder that includes batch normalization and a residual connection around the custom GAT layer.

    Args:
        node_input_dim (int): Dimension of input node features
        edge_input_dim (int): Dimension of input edge features
        hidden_dim (int): Hidden dimension for GAT convolutions
        dropout (float): Dropout rate for GAT convolutions
        layers (int): Number of GAT convolutional layers
        heads (int): Number of attention heads for each GAT convolution

    Returns:
        torch.Tensor: Graph-level embedding
    """

    def __init__(
        self, node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads
    ):
        super(ResidualEdgeGATEncoder, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.layers = layers
        # Fully connected layers for node and edge embedding

        self.fc_node = Linear(node_input_dim, hidden_dim * heads)
        self.fc_edge = Linear(edge_input_dim, hidden_dim * heads)
        # Output layer to reduce dimensionality of GAT output

        self.fc_out = Linear(hidden_dim * heads, hidden_dim)
        # Batch normalization layers for node and edge embedding

        self.bn_node = BatchNorm(hidden_dim * heads)
        self.bn_edge = BatchNorm(hidden_dim * heads)
        # List of EdgeGATConv modules with specified number of layers

        self.edge_gat_convs = torch.nn.ModuleList(
            [
                EdgeGATConv(
                    hidden_dim, hidden_dim, hidden_dim, dropout=dropout, heads=heads
                )
                for _ in range(layers)
            ]
        )
        # Initialize model parameters

        self.reset_parameters()

    def reset_parameters(self):
        """
        This function initializes the parameters of the encoder.
        """
        # torch.nn.init.xavier_uniform_(self.fc_node.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_edge.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_out.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_node.weight)
        torch.nn.init.xavier_uniform_(self.fc_edge.weight)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.constant_(self.fc_node.bias, 0)
        torch.nn.init.constant_(self.fc_edge.bias, 0)
        torch.nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, data: Any) -> Tuple[torch.Tensor]:
        """
        Forward pass for the ResidualEdgeGATEncoder.

        Args:
            data (Any): Graph data with node features, edge indices, and batch labels

        Returns:
            torch.Tensor: Node embedding
        """
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Extract input data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Node and edge embedding
        # Compute initial node and edge embeddings using fully connected layers

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

        return x  # , edge_attr, graph_embedding
