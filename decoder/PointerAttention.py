import torch  # Importing the PyTorch library
import torch.nn as nn  # Importing the PyTorch neural network module
import torch.nn.functional as F  # Importing the PyTorch functional API module
import math  # Importing the Python built-in `math` module for mathematical operations
from decoder.TransformerAttention import TransformerAttention  # Importing the custom TransformerAttention class
class PointerAttention(nn.Module):
    """
    This class represents a single head attention layer for the pointer network.

    Attributes:
        n_heads (int): The number of heads in the multi-head attention mechanism.
        input_dim (int): The dimensionality of the input vectors.
        hidden_dim (int): The dimensionality of the hidden vectors.
    """

    def __init__(self, n_heads, input_dim, hidden_dim):
        """
        Initializes a PointerAttention instance.
        Args:
            n_heads (int): The number of heads in the multi-head attention mechanism.
            input_dim (int): The dimensionality of the input vectors.
            hidden_dim (int): The dimensionality of the hidden vectors.
        """
        super(PointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Normalization factor for numerical stability in the softmax function
        self.norm = 1 / math.sqrt(hidden_dim)

        # Linear transformation for projecting input vectors to hidden space
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Multi-head attention layer instance
        self.mhalayer = TransformerAttention(n_heads, 1, input_dim, hidden_dim)

        # Initializes the model's weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the model's weights using Xavier initialization.
        
        This method is used to set the initial values of the network's parameters.
        In this case, we're using Xavier initialization for the weight matrices,
        and setting bias terms to zero.
        """
        for name, param in self.named_parameters():
            # Check if the parameter has more than one dimension (i.e., it's a weight matrix)
            if param.dim() > 1:
                # Apply Xavier initialization to the weight matrix
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # Check if the parameter is a bias term
                # Set the bias term to zero
                nn.init.constant_(param, 0)
    def forward(self, state_t, context, mask, T):
        """
        This function computes the attention scores, applies the mask, computes the nodes probabilities and returns them as a softmax score.
        - Applies a clipping to the attention scores to avoid numerical instability.

        Args:
            state_t (Tensor): The current state of the model.
            context (Tensor): The context vectors to attend to.
            mask (Tensor): The mask to apply to the attention scores.
            T (float): The temperature for the softmax function.

        Returns:
            Tensor: The softmax scores of the attention layer.
        """
        # Compute the attention scores using the multi-head attention mechanism
        x = self.mhalayer(state_t, context, mask)

        # Get the batch size and number of nodes from the context tensor
        batch_size, n_nodes, _ = context.size()
        
        # Project the state vector to hidden space using a linear transformation
        Q = x.reshape(batch_size, 1, -1)
        
        # Project the context vectors to hidden space using a linear transformation
        K = self.k(context).reshape(batch_size, n_nodes, -1)

        # Compute the compatibility scores between the projected state vector and the context vectors
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # Size: (batch_size, 1, n_nodes)
        
        # Remove the extra dimension from the compatibility scores
        compatibility = compatibility.squeeze(1)
        
        # Apply a scaling transformation to avoid numerical instability in the softmax function
        x = torch.tanh(compatibility)
        
        # Scale the values by multiplying with a scalar (10)
        x = x * (10)
        # Apply the mask to the attention scores
        x = x.masked_fill(mask.bool(), float("-inf"))
        
        # Compute the softmax scores using the masked attention scores and temperature T
        scores = F.softmax(x / T, dim=-1)
        
        return scores