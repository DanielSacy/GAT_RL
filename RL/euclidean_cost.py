import torch

def euclidean_cost(static, actions, batch):
    """
    Compute the total route cost based on the actions (node indices) and the static coordinates of the nodes.
    Args:
        static: torch.Tensor, shape [batch_size, num_nodes, 2], containing the (x, y) coordinates of each node.
        actions: torch.Tensor, shape [batch_size, num_nodes], containing the indices of the nodes in the route.
        num_nodes: int, the number of nodes in the problem (including the depot).
    """
    num_nodes = int(batch.x.size(0)/batch.batch_size)

    # Reshape and transpose the static coordinates to match the expected dimensions
    static = static.reshape(-1, num_nodes, 2)
    static = static.transpose(2, 1)

    # Get the coordinates of the nodes in the order they are visited according to actions
    idx = actions.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static, 2, idx).permute(0, 2, 1)

    # Add the depot at the start and end of the tour
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Compute the Euclidean distance between consecutive points
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    # Sum the distances to get the total tour length for each batch
    total_tour_len = tour_len.sum(1).detach()

    # Penalize depot visits
    depot_visits = (actions == 0).sum(dim=1)
    depot_penalty = 0.3 * (depot_visits)  # Subtract 1 to exclude the initial depot visit
    total_tour_len += depot_penalty

    return total_tour_len.detach()