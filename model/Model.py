import torch
from torch import nn
from src_batch.encoder.GAT_Encoder import ResidualEdgeGATEncoder
from src_batch.decoder.GAT_Decoder import GAT_Decoder


class Model(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity):
        super(Model, self).__init__()
        self.encoder = ResidualEdgeGATEncoder(node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads)
        self.decoder = GAT_Decoder(hidden_dim, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.capacity = capacity

    def forward(self, data,  n_steps, greedy, T=1):
        x = self.encoder(data)  # Shape of x: (n_nodes, hidden_dim) 
        
        batch_size = data.batch.max().item() + 1
        num_nodes_per_graph = data.num_nodes // batch_size
        
        x = x.reshape(batch_size, -1, self.hidden_dim) # Shape of x: (batch_size, n_nodes_per_graph, hidden_dim)?
        
        graph_embedding = x.mean(dim=1) # Shape of graph_embedding: (batch_size, hidden_dim)

        demand = data.demand.view(batch_size, -1).float().to(data.x.device)
        capacity = self.capacity.clone().detach().unsqueeze(-1).float().to(data.x.device)
        # capacity = torch.tensor(self.capacity).unsqueeze(-1).float().to(data.x.device)
        
        depot_visits = 0
        actions, log_p, depot_visits = self.decoder(x, graph_embedding, capacity, demand, n_steps,T, greedy, depot_visits)
        return actions, log_p, depot_visits