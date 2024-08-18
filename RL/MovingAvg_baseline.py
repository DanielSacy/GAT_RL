import torch

class MovingAverageBaseline:
    def __init__(self, beta=0.9):
        """
        Initialize the moving average baseline with a beta value for exponential smoothing.

        Args:
        - beta (float): Controls the weight of the previous baseline in the update rule. (0 < beta < 1)
        """
        self.beta = beta
        self.baseline = None

    def update(self, current_return):
        """
        Update the baseline using the exponential moving average formula.

        Args:
        - current_return (torch.Tensor): The return (reward) of the current batch or episode as a tensor.

        Returns:
        - updated_baseline (torch.Tensor): The updated moving average baseline as a tensor.
        """
        if self.baseline is None:
            # Initialize the baseline as the first return (must be a tensor)
            self.baseline = current_return
        else:
            # Update the baseline using the moving average formula
            self.baseline = self.beta * self.baseline + (1 - self.beta) * current_return

        return self.baseline

    def get_baseline(self):
        """
        Retrieve the current baseline as a tensor.

        Returns:
        - baseline (torch.Tensor): The current baseline tensor.
        """
        if self.baseline is None:
            return torch.tensor(0.0)
        return self.baseline
