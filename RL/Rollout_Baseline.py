import copy

from src_batch.RL.Compute_Reward import compute_reward

class RolloutBaseline:
    def __init__(self, model, n_rollouts):
        self.model = copy.deepcopy(model)
        self.n_rollouts = n_rollouts
    
    def rollout(self, data, n_steps):
        """
        Perform rollouts to compute the baseline.
        """
        self.model.eval()
        # bl_rewards = []
        # for _ in range(self.n_rollouts):
        actions, _, _ = self.model(data, n_steps, greedy=True, T=1)
        bl_rewards = compute_reward(actions, data)
        # bl_rewards.append(bl_reward)
        # print("bl_rewards:", bl_rewards)
        
        return bl_rewards