import torch
import networkx as nx

def mst_baseline(data):
    """
    Compute the MST baseline for the given data.
    """
    # Extract the data
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    node_features = data.x
    demand = data.demand
    capacity = data.capacity
    
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(range(node_features.size(0)))
    G.add_weighted_edges_from([(edge_index[0, i].item(), edge_index[1, i].item(), edge_attr[i].item()) for i in range(edge_index.size(1))])
    
    # Compute the MST
    mst = nx.minimum_spanning_tree(G)
    
    # Compute the MST route and value
    mst_route = []
    mst_value = 0
    for edge in mst.edges(data=True):
        mst_route.append(edge[0])
        mst_route.append(edge[1])
        mst_value += edge[2]['weight']
    
    # Compute the MST baseline
    mst_baseline_route = torch.tensor(mst_route, dtype=torch.long)
    mst_baseline_value = torch.tensor([mst_value], dtype=torch.float)
    
    return mst_baseline_route, mst_baseline_value