import torch


def compute_reward(actions, data):
    """
    Compute the reward for the given actions and data.
    args:
        actions: torch.Tensor, shape [batch_size, num_nodes]
        data: torch_geometric.data.Data
    """
    batch_size = actions.size(0)
    rewards = []

    # Edge index and attributes already on GPU
    edge_index = data.edge_index  # shape [2, num_edges]
    edge_attr = data.edge_attr.squeeze(1)  # shape [num_edges]

    # Create a dictionary for fast lookup of distances
    edge_dict = {(edge_index[0, i].item(), edge_index[1, i].item()): edge_attr[i] for i in range(edge_index.size(1))}

    for b in range(batch_size):
        route = actions[b].tolist()
        route_distance = 0

        for i in range(len(route) - 1):
            current_location = route[i]
            next_location = route[i + 1]
            
            # Fetch the precomputed distance
            route_distance += edge_dict.get((current_location, next_location), 0)
        
        # Store the route distance as reward
        # Should it be positive or negative?
        print(f'type(route_distance): {type(route_distance)}')
        rewards.append(route_distance)
    return torch.tensor(rewards, device=actions.device)

