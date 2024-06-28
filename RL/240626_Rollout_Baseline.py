import numpy as np

class RolloutBaseline:
    def __init__(self, model, n_rollouts=10):
        self.model = model
        self.n_rollouts = n_rollouts
    
    def rollout(self, data, n_steps):
        """
        Perform rollouts to compute the baseline.
        """
        rewards = []
        for _ in range(self.n_rollouts):
            actions, log_p, depot_visits = self.model(data, n_steps, greedy=True, T=1)
            reward = self.compute_reward(actions, data)
            rewards.append(reward)
        
        return np.mean(rewards)
    
    def compute_reward(self, actions, data):
        """
        Compute the CVRP reward based on the actions taken.
        """
        total_distance = 0
        batch_size = actions.size(0)
        
        edge_index = data.edge_index.t().cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        
        for b in range(batch_size):
            route = actions[b].cpu().numpy()
            route_distance = 0
            
            current_location = 0
            capacity_left = data.capacity[b].item()
            
            # Check if the route starts and ends at the depot
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
            
            total_distance += route_distance
        
        reward = -total_distance
        
        return reward
    
    def get_edge_distance(self, edge_index, edge_attr, current_location, next_location):
        """
        Get the distance for the edge between current_location and next_location.
        """
        for i, (from_node, to_node) in enumerate(edge_index):
            if from_node == current_location and to_node == next_location:
                return edge_attr[i].item()
        return 0  # Shouldn't reach here if the graph is fully connected