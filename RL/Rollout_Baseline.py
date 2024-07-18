import copy
import torch
from scipy.stats import ttest_rel
from src_batch.RL.Compute_Reward import compute_reward


# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout(model, dataset, n_steps, T):
    model.eval()
    def eval_model_bat(data):
        with torch.no_grad():
            actions, log_p, depot_visits = model(data, n_steps, greedy=True, T=T)
            reward = compute_reward(actions, data)
        return reward.cpu()

    total_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return total_cost

class RolloutBaseline():
    def __init__(self, model, dataset, n_steps, T,epoch=0):
        super(RolloutBaseline, self).__init__()
        self.n_steps = n_steps
        self.dataset = dataset
        self.T = T
        self._update_model(model, epoch)
        
    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        self.bl_vals = rollout(self.model, self.dataset, n_steps=self.n_steps).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, data, n_steps):
        with torch.no_grad():
            actions, log_p, depot_visits = self.model(data, n_steps, greedy=True, T=self.T)
            reward = compute_reward(actions, data)
        return reward

    def epoch_callback(self, model, epoch):
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.n_steps).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        load_model.load_state_dict(state_dict['model']).state_dict()
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])