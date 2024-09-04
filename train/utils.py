import torch
from ..RL.euclidean_cost_eval import euclidean_cost_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Validation function
def evaluate_on_validation(actor, validation_loader, n_steps, T):
    actor.eval()  # Set model to evaluation mode
    num_samples = 1
    validation_rewards = []
    
    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(device)
            
            # Run greedy sampling on validation data
            actions_list, _ = actor(batch, n_steps, greedy=True, T=T, num_samples=num_samples)
            actions = torch.tensor(actions_list[0], dtype=torch.long, device=device)
                        
            # Compute the cost of the actions
            cost = euclidean_cost_eval(batch.x, actions, batch)  # Shape: (batch_size,)
            
            validation_rewards.append(cost.mean().item())
    
    # Calculate the average reward over the validation set
    mean_validation_reward = sum(validation_rewards) / len(validation_rewards)
    return mean_validation_reward