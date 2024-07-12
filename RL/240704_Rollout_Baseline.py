import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from scipy.stats import ttest_rel

class RolloutBaseline:
    def __init__(self, model, data_loader, n_steps, n_nodes=50, device='cuda', warmup_beta=0.8, update_freq=5, epoch=0):
        self.model = model
        self.data_loader = data_loader
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self.device = device
        self.warmup_beta = warmup_beta
        self.update_freq = update_freq
        self.alpha = 0.0  # Baseline adaptation rate
        self.M = None  # Initialize the moving average for EMA
        self._update_baseline(model, epoch)

    def _update_baseline(self, model, epoch):
        self.model = self.copy_model(model)
        self.model.to(self.device)
        print(f'Evaluating baseline model on baseline dataset (epoch = {epoch})')
        self.bl_vals = self.rollout(self.model, self.data_loader).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def copy_model(self, model):
        new_model = copy.deepcopy(model)
        return new_model

    def rollout(self, model, data_loader):
        model.eval()
        costs_list = []
        for data in tqdm(data_loader, desc='Rollout greedy execution'):
            data = data.to(self.device)
            with torch.no_grad():
                actions, _, _ = model(data, self.n_steps, greedy=True, T=1)
                reward = self.compute_reward(actions, data)
                costs_list.append(reward)
        return torch.stack(costs_list)

    def compute_reward(self, actions, data):
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
            
            if route[0] != 0 and route[-1] != 0:
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

            total_distance += route_distance
        reward = -total_distance
        return torch.tensor([reward], dtype=torch.float32, device=self.device)

    def get_edge_distance(self, edge_index, edge_attr, current_location, next_location):
        for i, (from_node, to_node) in enumerate(edge_index):
            if from_node == current_location and to_node == next_location:
                return edge_attr[i].item()
        return 0  # Shouldn't reach here if the graph is fully connected

    def eval(self, data, reward):
        if self.alpha == 0:
            return self.ema_eval(reward)
        if self.alpha < 1:
            v_ema = self.ema_eval(reward)
        else:
            v_ema = 0.0
        with torch.no_grad():
            v_b, _, _ = self.model(data, self.n_steps, greedy=True, T=1)
            v_b = self.compute_reward(v_b, data)
        return self.alpha * v_b + (1 - self.alpha) * v_ema

    def ema_eval(self, reward):
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device) if not isinstance(reward, torch.Tensor) else reward
        if self.M is None:
            self.M = reward_tensor.mean()
        else:
            self.M = self.warmup_beta * self.M + (1. - self.warmup_beta) * reward_tensor.mean()
        return self.M.detach()

    def epoch_callback(self, model, epoch):
        if epoch % self.update_freq == 0:
            self._update_baseline(model, epoch)
        self.alpha = min(1, (epoch + 1) / float(self.update_freq))
