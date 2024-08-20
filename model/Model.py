import torch
from torch import nn
from src_batch.encoder.GAT_Encoder import ResidualEdgeGATEncoder
from src_batch.decoder.GAT_Decoder import GAT_Decoder


class Model(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope,dropout):
        super(Model, self).__init__()
        self.encoder = ResidualEdgeGATEncoder(node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout)
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)

    def forward(self, data,  n_steps, greedy, T):
        # data.x: node features, data.edge_attr: edge features, data.edge_index: edge indices
        x = self.encoder(data)  # Shape of x: (n_nodes, hidden_dim) 
        # Compute the graph embedding > mean of all node embeddings per feature dimension
        graph_embedding = x.mean(dim=1) # Shape of graph_embedding: (batch_size, hidden_dim)

        # Get the demand and capacity - Detach them?
        batch_size = data.batch.max().item() + 1
        demand = data.demand.reshape(batch_size, -1).float().to(data.x.device)
        capacity = data.capacity.reshape(batch_size, -1).float().to(data.x.device)
        
        # Call the decoder
        actions, log_p = self.decoder(x, graph_embedding, capacity, demand, n_steps,T, greedy)
        
        return actions, log_p