import torch
from torch import optim
from src_batch.train.beam_search import beam_search

class Train:
    def __init__(self, model, data_loader, device, baseline, n_steps, optimizer=None, lr=0.001, beam_width=1):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.baseline = baseline
        self.n_steps = n_steps
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=lr)
        self.beam_width = beam_width
        
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            for data in self.data_loader:
                data = data.to(self.device)
                
                greedy = True
                if greedy == False:
                    actions, log_p = beam_search(self.model, data, self.beam_width, self.n_steps)
                else:
                    actions, log_p, _ = self.model(data, self.n_steps, greedy=False, T=1)
                
                rewards = []
                baseline_rewards = []
                advantages = []

                for action, log_prob in zip(actions, log_p):
                    reward = self.baseline.compute_reward(action.unsqueeze(0), data)
                    baseline_reward = self.baseline.eval(data, reward)
                    advantage = reward - baseline_reward

                    rewards.append(reward)
                    baseline_rewards.append(baseline_reward)
                    advantages.append(advantage)

                    loss = -(log_prob * advantage).mean()
                
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                avg_loss = torch.stack([-(lp * adv).mean() for lp, adv in zip(log_p, advantages)]).mean()
                avg_reward = torch.stack(rewards).mean().item()
                avg_baseline_reward = torch.stack(baseline_rewards).mean().item()
                avg_advantage = torch.stack(advantages).mean().item()

                print(f'Epoch {epoch:<5}, Loss: {avg_loss:<8.3f}, Reward: {avg_reward:<10}, Baseline: {avg_baseline_reward:<10}, Advantage: {avg_advantage:<10}')

            self.baseline.epoch_callback(self.model, epoch)
