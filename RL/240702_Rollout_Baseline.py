import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

class RolloutBaseline:
    def __init__(self, model, data_loader, n_steps, n_rollouts=10, device='cuda'):
        self.model = model
        self.data_loader = data_loader
        self.n_steps = n_steps
        self.n_rollouts = n_rollouts
        self.device = device
        self._update_baseline(model, 0)

    def _update_baseline(self, model, epoch):
        self.model = self.copy_model(model)
        self.model.to(self.device)
        print(f'Evaluating baseline model on baseline dataset (epoch = {epoch})')
        self.bl_vals = self.rollout(self.model, self.data_loader).cpu().numpy()
        self.mean = self.bl_vals.mean()

    def copy_model(self, model):
        new_model = copy.deepcopy(model)
        return new_model

    def rollout(self, model, data_loader):
        costs_list = []
        for data in data_loader:
        # for data in tqdm(data_loader, desc='Rollout greedy execution'):
            data = data.to(self.device)
            with torch.no_grad():
                actions, log_p, depot_visits = model(data, self.n_steps, greedy=True, T=1)
                cost = self.compute_reward(actions, data)
                costs_list.append(cost)

        return torch.stack(costs_list)

    def eval(self, data, cost):
        return self.rollout(self.model, self.data_loader).mean()
    
    def epoch_callback(self, model, epoch):
        self._update_baseline(model, epoch)
