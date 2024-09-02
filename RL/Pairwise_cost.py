import torch
import itertools

def pairwise_cost(actions, batch):
    """
    Compute the reward for the given actions and precomputed distance matrix.
    Args:
        actions: torch.Tensor, shape [batch_size, num_nodes]
        batch: Data object containing a precomputed distance matrix (batch.distance_matrix)
               distance_matrix shape: [batch_size, num_nodes, num_nodes]
    """
    batch_size = actions.size(0)
    num_nodes = actions.size(1)

    # Add depot (node 0) to the start and end of each route
    depot = torch.zeros(batch_size, 1, dtype=torch.long, device=actions.device)
    routes = torch.cat([depot, actions, depot], dim=1)  # Shape: [batch_size, num_nodes + 2]

    # Access the distance_matrix from the batch object
    distance_matrix = batch.distance_matrix  # Shape: [batch_size, num_nodes, num_nodes]

    # Shift the route to the right by 1 to create consecutive pairs of nodes
    current_nodes = routes[:, :-1]  # Shape: [batch_size, num_nodes + 1]
    next_nodes = routes[:, 1:]      # Shape: [batch_size, num_nodes + 1]

    # Gather distances for the consecutive pairs in the routes
    # Use advanced indexing to get the distances for the route transitions
    route_distances = distance_matrix[current_nodes, next_nodes]  # Shape: [batch_size, num_nodes + 1]

    # Sum the route distances to get the total distance for each route
    total_route = route_distances.sum(dim=1)  # Shape: [batch_size]
    
    # Count the number of depot visits
    depot_visits = (routes == 0).sum(dim=1) - 3  # Exclude start and end depot visits

    # Apply penalty for depot visits
    depot_penalty = 0.3 * depot_visits  # Modify this factor as needed

    total_distances = total_route + depot_penalty

    # Return the negative route distances as rewards (cost minimization problem)
    return total_distances.detach()

# def pairwise_cost(actions, batch):
#     """
#     Compute the reward for the given actions and precomputed distance matrix.
#     args:
#         actions: torch.Tensor, shape [batch_size, num_nodes]
#         distance_matrix: torch.Tensor, shape [num_nodes, num_nodes] - Precomputed distance matrix
#     """
#     batch_size = actions.size(0)
#     costs = []
    
#     # Access the distance_matrix from the Data object
#     distance_matrix = batch.distance_matrix  # Tensor of shape [num_nodes, num_nodes]
#     depot = torch.tensor([0]).to(actions.device)

#     # Loop through each batch
#     for b in range(batch_size):
#         route = actions[b]  # Tensor of shape [num_nodes]
#         # Add depot to the start and end of the route
#         route = torch.cat([depot, route, depot])

#         # Initialize route distance to zero
#         route_distance = 0

#         # Iterate over consecutive pairs of nodes in the route using itertools.pairwise
#         for current_node, next_node in itertools.pairwise(route):
#             route_distance += distance_matrix[current_node, next_node]

#         # Append the negative route distance as the reward
#         costs.append(route_distance)

#     # Convert the rewards list to a tensor and return it
#     return torch.tensor(costs, device=actions.device).detach()