import copy
import torch
from scipy.stats import ttest_rel
from ..RL.euclidean_cost import euclidean_cost


# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rollout(model, dataset, n_nodes, T):
    model.eval()  # Set the model to evaluation mode

    def eval_model_bat(batch):
        with torch.no_grad():  # No gradients during evaluation
            # Simulate the model on the batchch of instances
            tour, _ = model(batch, n_nodes * 2, greedy=True, T=T)
            # Compute the cost of the tour using the reward function
            cost = euclidean_cost(batch.x, tour.detach(), batch)
        return cost

    # Concatenate all the results across the dataset
    total_cost = torch.cat([eval_model_bat(batch.to(device)) for batch in dataset], dim=0)
    return total_cost

class RolloutBaseline():
    def __init__(self, model, dataset, n_nodes=20, epoch=0, T=1.0):
        super(RolloutBaseline, self).__init__()
        self.n_nodes = n_nodes  # Number of nodes in the VRP/TSP
        self.dataset = dataset  # Dataset to generate rollouts
        self.T = T  # Temperature for the softmax
        self._update_model(model, epoch)  # Initialize the baseline with the current model

    def _update_model(self, model, epoch):
        """Deepcopy the model and compute baseline values"""
        self.model = copy.deepcopy(model).to(device)  # Deepcopy the model
        self.bl_vals = rollout(self.model, self.dataset, self.n_nodes, self.T).cpu().numpy()  # Compute the baseline values
        self.mean = self.bl_vals.mean()  # Calculate the mean of the baseline values
        self.epoch = epoch

    def eval(self, batch, n_nodes):
        """Evaluate the baseline model on a batch of data"""
        with torch.no_grad():
            tour, _ = self.model(batch, n_nodes, greedy=True, T=self.T)
            base = euclidean_cost(batch.x, tour.detach(), batch)
        return base.detach()

    def epoch_callback(self, model, epoch):
        """Evaluate the new model and compare it with the baseline"""
        print("Evaluating candidate model on evaluation dataset")

        candidate_vals = rollout(model, self.dataset, self.n_nodes, self.T).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        if candidate_mean < self.mean:
            # Perform a paired t-test to check if the new model is significantly better
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # One-sided t-test
            assert t < 0, "T-statistic should be negative"

            print(f"p-value: {p_val}")
            if p_val < 0.05:
                print("Update baseline")
                self._update_model(model, epoch)  # Update the baseline if the new model is better

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        """Load the baseline state from a saved checkpoint"""
        load_model = copy.deepcopy(self.model)
        load_model.load_state_dict(state_dict['model'].state_dict())
        self._update_model(load_model, state_dict['epoch'])