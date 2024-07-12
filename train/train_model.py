# import torch
# from torch import optim

# class Train:
#     def __init__(self, model, data_loader, device, baseline, n_steps, optimizer=None, lr=0.001):
#         self.model = model
#         self.data_loader = data_loader
#         self.device = device
#         self.baseline = baseline
#         self.n_steps = n_steps
#         self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=lr)
        
#     def train(self, n_epochs):
#         self.model.to(self.device)
        
#         for epoch in range(n_epochs):
#             for data in self.data_loader:
#                 data = data.to(self.device)
                
#                 # Policy network
#                 self.model.train()
#                 actions, log_p, _ = self.model(
#                     data, self.n_steps, greedy=False, T=1#2.5
#                     )
                
#                 # Compute reward and baseline
#                 reward = self.baseline.compute_reward(actions, data)
                
#                 self.model.eval()
#                 with torch.no_grad():
#                     baseline_actions, _, _ = self.model(
#                     data, self.n_steps, greedy=True, T=1#2.5
#                     )
#                     baseline_reward = self.baseline.compute_reward(baseline_actions, data)
                
#                 print(f'Baseline reward: {baseline_reward}')
                
#                 advantage = reward - baseline_reward
                
#                 loss = -(log_p * advantage).mean()
                
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 self.optimizer.step()
                
#                 print(f'Epoch {epoch:<5}, Loss: {loss:<8.3f}, Reward: {reward:<10.3f}, Baseline: {baseline_reward:<10.3f}, Advantage: {advantage:<10.3f}')

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
                self.model.train()
                actions, log_p, _ = self.model(data, self.
                                n_steps, greedy=False, T=1)
                
                # Compute reward and baseline
                reward = self.baseline.compute_reward(actions, data)
                baseline_reward = self.baseline.rollout(data, self.n_steps)
                # print(f'Baseline reward: {baseline_reward}')
                
                advantage = reward - baseline_reward
                
                loss = -(log_p * advantage).mean()
                # loss = -log_p.mean() * advantage
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                print(f'Epoch {epoch:<5}, Loss: {loss:<8.3f}, Reward: {reward:<10}, Baseline: {baseline_reward:<10}, Advantage: {advantage:<10}')
