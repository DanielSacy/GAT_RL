import torch
from torch.nn import Linear, BatchNorm1d as BatchNorm, Sequential, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, scatter
import torch.nn.functional as F

# Custom GAT layer that includes edge features in the computation of attention coefficients.
class EdgeGATConv(MessagePassing):
    """
    This class is a custom GAT layer that includes edge features in the computation of attention coefficients.
    It also uses multi-head attention to improve the performance of the model.
    """
    def __init__(self, node_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0.6, heads=1, concat=True):
        super(EdgeGATConv, self).__init__(aggr='add' if concat else 'mean') 
        self.node_channels = node_channels
        self.out_channels = out_channels  # Number of output channels per head
        self.edge_channels = edge_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.heads = heads
        self.concat = concat
        
        self.fc = torch.nn.Linear(out_channels * 3, out_channels, bias=False)
        self.att_vector = torch.nn.Parameter(torch.Tensor(1, out_channels))
        # self.out_channel_linear = Linear(out_channels * heads, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.att_vector.data)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, edge_index, edge_attr, size=None):
        """This function computes the node embeddings."""
                
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
    
    def message(self, edge_index_i, x_i, x_j, edge_attr, size_i):
        """This function computes the attention coefficients and returns the message to be aggregated."""
        # Reshape to separate heads for independent attention 
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.edge_channels)
        
        # Concatenate features
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute attention coefficients
        alpha = self.fc(x)
        alpha = torch.matmul(alpha, self.att_vector.T)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) #Dropout works by setting some neurons to zero with probability p and scaling the remaining neurons by 1/(1 - p) to avoid overfitting.
        return x_j * alpha

    def update(self, aggr_out):
        """This function updates the node embeddings."""
        return aggr_out
    
    def aggregate(self, inputs, index):
        """This function aggregates the messages"""
        aggr = scatter(inputs, index, dim=0, reduce=self.aggr)
        
        num_nodes = aggr.size(0)
        aggr = aggr.view(num_nodes, -1) # Concatenates the heads        
        # aggr = self.out_channel_linear(aggr) # Corrects the output shape
        return aggr