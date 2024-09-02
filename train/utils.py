import torch


def scale_to_range(cost, min_val=0.0, max_val=1.0):
    # Compute the minimum and maximum values of the cost
    min_cost = cost.min()
    max_cost = cost.max()
    
    # Scale the cost to the [0, 1] range
    scaled_cost = (cost - min_cost) / (max_cost - min_cost + 1e-8)
    
    # Scale it to the desired range [min_val, max_val]
    # scaled_cost = scaled_cost * (max_val - min_val) + min_val
    
    return scaled_cost, min_cost, max_cost

def scale_back(scaled_cost, min_cost, max_cost):
    # Scale the cost back to the original range
    original_cost = scaled_cost * (max_cost - min_cost) + min_cost
    return original_cost

def normalize(values):
    std = values.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_values = (values - values.mean()) / (values.std() + 1e-8)
    return n_values