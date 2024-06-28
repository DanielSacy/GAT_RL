import torch
from torch import optim

class Train:
    def __init__(self, model, data_loader, device, baseline, n_steps, optimizer=None, lr=0.001):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.baseline = baseline
        self.n_steps = n_steps
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=lr)
        
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            for data in self.data_loader:
                data = data.to(self.device)
                
                # Policy network
                actions, log_p, _ = self.model(data, self.
                                                          n_steps, greedy=True, T=1)
                
                # Compute reward and baseline
                reward = self.baseline.compute_reward(actions, data)
                baseline_reward = self.baseline.rollout(data, self.n_steps)
                
                
                advantage = reward - baseline_reward
                
                loss = -(log_p * advantage).mean()
                # loss = -log_p.mean() * advantage
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                print(f'Epoch {epoch:<5}, Loss: {loss:<8.3f}, Reward: {reward:<10}, Baseline: {baseline_reward:<10}, Advantage: {advantage:<10}')
                
                
# import torch
# from torch import optim

# class Train:
#     def __init__(self, model, data_loader, device, baseline, n_steps, optimizer=None, lr=0.001, gamma=0.99):
#         self.model = model
#         self.data_loader = data_loader
#         self.device = device
#         self.baseline = baseline
#         self.n_steps = n_steps
#         self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=lr)
#         self.gamma = gamma  # Discount factor
    
#     def compute_discounted_rewards(self, rewards, gamma=0.99):
#         discounted_rewards = []
#         cumulative_reward = 0
#         for reward in reversed(rewards):
#             cumulative_reward = reward + gamma * cumulative_reward
#             discounted_rewards.insert(0, cumulative_reward)
#         return discounted_rewards
    
#     def train(self, n_epochs):
#         for epoch in range(n_epochs):
#             epoch_rewards = []
#             epoch_baseline_rewards = []
#             epoch_log_probs = []

#             for data in self.data_loader:
#                 data = data.to(self.device)
                
#                 # Rollout the baseline to get the baseline reward
#                 baseline_reward = self.baseline.rollout(data, self.n_steps)
                
#                 # Sample actions from the model
#                 actions, log_p, _ = self.model(data, self.n_steps, greedy=True, T=1)
                
#                 # Check if log_p is calculated correctly
#                 if log_p is None:
#                     print("Warning: log_p is None")
#                     continue
                
#                 # Compute the actual reward
#                 reward = self.baseline.compute_reward(actions, data)
                
#                 # Store rewards and log probabilities
#                 epoch_rewards.append(reward)
#                 epoch_baseline_rewards.append(baseline_reward)
#                 epoch_log_probs.append(log_p)
            
#             # Verify collected log probabilities
#             if not epoch_log_probs:
#                 print("Error: No log probabilities collected during the epoch.")
#                 continue

#             # Convert lists to tensors for processing
#             epoch_rewards_tensor = torch.tensor(epoch_rewards, dtype=torch.float32).to(self.device)
#             epoch_baseline_rewards_tensor = torch.tensor(epoch_baseline_rewards, dtype=torch.float32).to(self.device)
#             epoch_log_probs_tensor = torch.cat(epoch_log_probs).to(self.device)
            
#             # Compute discounted rewards
#             discounted_rewards = self.compute_discounted_rewards(epoch_rewards_tensor.tolist(), self.gamma)
#             discounted_baseline = self.compute_discounted_rewards(epoch_baseline_rewards_tensor.tolist(), self.gamma)
            
#             # Convert to tensors for normalization
#             discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
#             discounted_baseline_tensor = torch.tensor(discounted_baseline, dtype=torch.float32).to(self.device)
            
#             # Normalize rewards
#             mean_reward = discounted_rewards_tensor.mean()
#             std_reward = discounted_rewards_tensor.std()
#             normalized_rewards = (discounted_rewards_tensor - mean_reward) / (std_reward + 1e-9)
            
#             # Normalize baseline rewards
#             mean_baseline = discounted_baseline_tensor.mean()
#             std_baseline = discounted_baseline_tensor.std()
#             normalized_baseline = (discounted_baseline_tensor - mean_baseline) / (std_baseline + 1e-9)
            
#             # Compute advantages
#             advantages = normalized_rewards - normalized_baseline
            
#             # Ensure advantage is finite
#             if not torch.isfinite(advantages).all():
#                 print(f"Warning: Encountered non-finite advantage values: {advantages}")
#                 continue
            
#             # Compute policy loss
#             policy_loss = -(epoch_log_probs_tensor * advantages).mean()
            
#             # Perform backpropagation and optimization step
#             self.optimizer.zero_grad()
#             policy_loss.backward()
#             self.optimizer.step()
            
#             # Print the current status
#             print(f'Epoch {epoch:<5}, Loss: {policy_loss:<8.3f}, Reward: {epoch_rewards_tensor.mean():<10.3f}, Baseline: {epoch_baseline_rewards_tensor.mean():<10.3f}, Advantage: {advantages.mean().item():<10.3f}')
