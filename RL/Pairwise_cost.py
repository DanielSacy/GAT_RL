import torch
import itertools

def pairwise_cost(actions, batch):
    """
    Compute the reward for the given actions and precomputed distance matrix.
    args:
        actions: torch.Tensor, shape [batch_size, num_nodes]
        distance_matrix: torch.Tensor, shape [num_nodes, num_nodes] - Precomputed distance matrix
    """
    batch_size = actions.size(0)
    costs = []
    
    # Access the distance_matrix from the Data object
    distance_matrix = batch.distance_matrix  # Tensor of shape [num_nodes, num_nodes]
    depot = torch.tensor([0]).to(actions.device)

    # Loop through each batch
    for b in range(batch_size):
        route = actions[b]  # Tensor of shape [num_nodes]
        # Add depot to the start and end of the route
        route = torch.cat([depot, route, depot])

        # Initialize route distance to zero
        route_distance = 0

        # Iterate over consecutive pairs of nodes in the route using itertools.pairwise
        for current_node, next_node in itertools.pairwise(route):
            route_distance += distance_matrix[current_node, next_node]

        # Append the negative route distance as the reward
        costs.append(route_distance)

    # Convert the rewards list to a tensor and return it
    return torch.tensor(costs, device=actions.device).detach()