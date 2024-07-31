import torch


def get_edge_distance(edge_index, edge_attr, current_location, next_location):
    """
    Get the distance for the edge between current_location and next_location.
    args:
        edge_index: torch.Tensor, shape [2, num_edges]
        edge_attr: torch.Tensor, shape [num_edges, 1]
        current_location: int
        next_location: int
    """
    for i, (from_node, to_node) in enumerate(edge_index):
        if from_node == current_location and to_node == next_location:
            # print(f'Edge from {from_node} to {to_node}, distance: {edge_attr[i].item()}\n')
            return edge_attr[i].item()
    return 0  # Shouldn't reach here if the graph is fully connected

def compute_reward(actions, data):
    """
    Compute the reward for the given actions and data.
    args:
        actions: torch.Tensor, shape [batch_size, num_nodes]
        data: torch_geometric.data.Data
    """
    # total_distance = 0
    rewards = []
    batch_size = actions.size(0)
    
    edge_index = data.edge_index.t().cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    
    for b in range(batch_size):
        route = actions[b].cpu().numpy()
        route_distance = 0
        # capacity_left = data.capacity[b].item()

        # # Check if the route starts and ends at the depot
        # if route[0] != 0:
        #     route_distance += 1000  # penalty for not starting at the depot
    
        # if route[-1] != 0:
        #     route_distance += 1000  # penalty for not ending at the depot

        for i in range(len(actions[b])):
            current_location = route[i].item()
            if i < len(route) - 1:
                next_location = route[i + 1].item()
            else:
                break

            route_distance += get_edge_distance(edge_index, edge_attr, current_location, next_location)
            
            # capacity_left -= data.demand[current_location]#data.demand[next_location]
            current_location = next_location
            
            # if capacity_left < 0:
            #     route_distance += 500  # penalty for exceeding capacity
        
        # total_distance += route_distance
        rewards.append(route_distance)
    
    # reward = -total_distance
    
    return torch.tensor(rewards, device=actions.device)