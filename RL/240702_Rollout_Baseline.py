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

    def compute_reward(self, actions, data):
        """
        Compute the CVRP reward based on the actions taken.
        """
        total_distance = 0
        batch_size = actions.size(0)
        
        edge_index = data.edge_index.t().cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        
        rewards = []
        for b in range(batch_size):
            route = actions[b].cpu().numpy()
            route_distance = 0
            
            current_location = 0
            capacity_left = data.capacity[b].item()
            
            if route[0] != 0 or route[-1] != 0:
                route_distance += 300  # penalty for not starting/ending at the depot

            for step in route:
                next_location = step.item()
                if next_location == 0:
                    route_distance += self.get_edge_distance(edge_index, edge_attr, current_location, next_location)
                    current_location = 0
                    capacity_left = data.capacity[b].item()
                else:
                    route_distance += self.get_edge_distance(edge_index, edge_attr, current_location, next_location)
                    capacity_left -= data.demand[next_location]
                    current_location = next_location

                if capacity_left < 0:
                    route_distance += 300  # penalty for exceeding capacity

            rewards.append(-route_distance)  # negative reward for minimizing distance

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def get_edge_distance(self, edge_index, edge_attr, current_location, next_location):
        for i, (from_node, to_node) in enumerate(edge_index):
            if from_node == current_location and to_node == next_location:
                return edge_attr[i].item()
        return 0  # Shouldn't reach here if the graph is fully connected

    def eval(self, data, cost):
        return self.rollout(self.model, self.data_loader).mean()
    
    def epoch_callback(self, model, epoch):
        self._update_baseline(model, epoch)
