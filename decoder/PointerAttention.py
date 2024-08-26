import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src_batch.decoder.TransformerAttention import TransformerAttention

class PointerAttention(nn.Module):
    """
    This class is the single head attention layer for the pointer network.
    """
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(PointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = TransformerAttention(n_heads, 1, input_dim, hidden_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """
        This function initializes the parameters of the encoder.
        It's using the Xavier initialization over Orthogonal initialization because it's more suitable for the ReLU activation function applied to the output of the attention layer.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Typically applies to weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # Check if it's a bias term
                nn.init.constant_(param, 0)  # Initialize biases to zero


    def forward(self, state_t, context, mask, T):
        '''
        This function computes the attention scores, applies the mask, computes the nodes probabilities and returns them as a softmax score.
        - Applies a clipping to the attention scores to avoid numerical instability.
        
        Args:
        - state_t: The current state of the model. (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        - context: The context to attend to. (batch_size,n_nodes,input_dim)
        - mask: The mask to apply to the attention scores. (batch_size,n_nodes)
        - T: The temperature for the softmax function.
        
        returns:
        - softmax_score: The softmax scores of the attention layer. (batch_size, n_nodes)
        '''
        x = self.mhalayer(state_t, context, mask)

        batch_size, n_nodes, input_dim = context.size()
        Q = x.reshape(batch_size, 1, -1)
        K = self.k(context).reshape(batch_size, n_nodes, -1)
        
        # Compute the compatibility scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # Size: (batch_size, 1, n_nodes)
        compatibility = compatibility.squeeze(1)
        # compatibility = compatibility.squeeze(1)/1000
        # Normalize compatibility scores for numerical stability in the softmax
        # compatibility = (compatibility - compatibility.mean()) / (compatibility.std() + 1e-8)

        # Non-linear transformation
        x = torch.tanh(compatibility)
        
        # Scaling the values to avoid numerical instability
        x = x * (10)
        
        # Apply the mask
        # x = compatibility.masked_fill(mask.bool(), float("-inf"))
        x = x.masked_fill(mask.bool(), float("-inf"))
        
        # Compute the softmax scores
        scores = F.softmax(x / T, dim=-1)
        return scores