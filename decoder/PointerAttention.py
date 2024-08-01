import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src_batch.decoder.TransformerAttention import TransformerAttention


class PointerAttention(nn.Module):
    """
    This class is the single head attention layer for the pointer network.

    Attributes:
        n_heads (int): The number of attention heads.
        input_dim (int): The dimensionality of the input.
        hidden_dim (int): The dimensionality of the hidden state.
    """

    def __init__(self, n_heads, input_dim, hidden_dim) -> None:
        super(PointerAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Normalize the weights for the attention scores
        self.norm = 1 / math.sqrt(hidden_dim)
        # Define the linear layer to transform the context into the hidden state dimensionality
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        # Initialize the TransformerAttention layer with a single head and an output dimensionality of 1
        self.mhalayer = TransformerAttention(n_heads, 1, input_dim, hidden_dim)
        # if INIT:
        #     for name, p in self.named_parameters():
        #         if 'weight' in name:
        #             if len(p.size()) >= 2:
        #                 nn.init.orthogonal_(p, gain=1)
        #         elif 'bias' in name:
        #             nn.init.constant_(p, 0)

        # Reset the parameters for this attention layer (using Xavier initialization instead of orthogonal initialization due to ReLU activation function)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        This function initializes the parameters of the attention layer.
        It's using the Xavier initialization over Orthogonal initialization because it's more suitable for the ReLU activation function applied to the output of the attention layer.
        """
        nn.init.xavier_uniform_(self.k.weight.data)

    def forward(
        self, state_t: torch.Tensor, context: torch.Tensor, mask: torch.Tensor, T: float
    ) -> torch.Tensor:
        """
        This function computes the attention scores, applies the mask, computes the nodes probabilities and returns them as a softmax score.
        - Applies a clipping to the attention scores to avoid numerical instability.

        Args:
        - state_t: The current state of the model. (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        - context: The context to attend to. (batch_size,n_nodes,input_dim)
        - mask: The mask to apply to the attention scores. (batch_size,n_nodes)
        - T: The temperature for the softmax function.

        returns:
        - softmax_score: The softmax scores of the attention layer. (batch_size, n_nodes)
        """
        # Apply the TransformerAttention layer to compute the compatibility between state_t and context
        x = self.mhalayer(state_t, context, mask)
        # Transform the input dimensionality of context into the hidden state dimensionality using the linear layer k
        batch_size, n_nodes, input_dim = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        # Compute the attention scores by taking the dot product of Q and K^T (transposed) and normalizing them using the norm attribute
        compatibility = self.norm * torch.matmul(
            Q, K.transpose(1, 2)
        )  # (batch_size,1,n_nodes)
        compatibility = compatibility.squeeze(1)
        # Clip the attention scores to avoid numerical instability
        x = torch.tanh(compatibility)
        x = x * (10)
        # min_value = torch.finfo(x.dtype).min
        # min_value = 1e-15
        # x = x.masked_fill(mask.bool(), min_value)

        # Mask the attention scores based on the mask tensor and apply a softmax function to them with temperature T

        x = x.masked_fill(mask.bool(), float("-inf"))

        # Compute the softmax scores
        scores = F.softmax(x / T, dim=-1)
        return scores
