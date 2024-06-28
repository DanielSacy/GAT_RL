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
            epoch_rewards = []
            epoch_baseline_rewards = []
            epoch_log_probs = []
            
            for data in self.data_loader:
                data = data.to(self.device)
                
                # Policy network
                actions, log_p, _ = self.model(data, self.
                                    n_steps, greedy=True, T=1)
                
                # Compute reward and baseline
                reward = self.baseline.compute_reward(actions, data)
                baseline_reward = self.baseline.eval(data, self.n_steps)
                
                
                advantage = reward - baseline_reward
                
                # Accumulate rewards and log probabilities
                epoch_rewards.append(reward)
                epoch_baseline_rewards.append(baseline_reward)
                epoch_log_probs.append(log_p)
                
                loss = -(log_p * advantage).mean()
                # loss = -log_p.mean() * advantage
                
                # Training step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
                self.optimizer.step()
            
                # print(f'Epoch {epoch:<5}, Loss: {loss:<8.3f}, Reward: {reward:<10}, Baseline: {baseline_reward:<10}, Advantage: {advantage:<10}')
                
            # Logging and checkpointing
            epoch_rewards_tensor = torch.tensor(epoch_rewards, dtype=torch.float32).to(self.device)
            epoch_baseline_rewards_tensor = torch.tensor(epoch_baseline_rewards, dtype=torch.float32).to(self.device)
            epoch_log_probs_tensor = torch.cat(epoch_log_probs).to(self.device)
            
            # Print epoch status
            print(f'Epoch {epoch:<5}, Loss: {loss:<8.3f}, Reward: {epoch_rewards_tensor.mean():<10.3f}, Baseline: {epoch_baseline_rewards_tensor.mean():<10.3f}, Advantage: {(epoch_rewards_tensor - epoch_baseline_rewards_tensor).mean().item():<10.3f}')
            
            # Save model checkpoint
            if epoch == n_epochs - 1:
                torch.save(self.model.state_dict(), f'model_checkpoints/model_checkpoint_epoch_{epoch}.pt')
            
            # Update the baseline
            self.baseline.epoch_callback(self.model, epoch)
                
                
