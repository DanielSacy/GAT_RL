import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerAttention(nn.Module):
    """
    This class computes the attention scores and returns the output.
    
    Args:
        n_heads (int): The number of heads.
        input_dim (int): The dimension of the input.
        hidden_dim (int): The hidden dimension for the output.
        attn_dropout (float, optional): The dropout rate for the attention scores. Defaults to 0.1.
        dropout (float, optional): The dropout rate for the output. Defaults to 0.
    Returns:
        out_put: The output of the attention layer.
    """

    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(TransformerAttention, self).__init__()
        """
        Assert that the hidden dimension is divisible by the number of heads.
        """
        assert hidden_dim % n_heads == 0, "Hidden dimension should be a multiple of the number of heads."

        # Define the parameters
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        # Initialize the fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize the weights and biases
        self.initialize_weights()

    def initialize_weights(self):
        """
        This function initializes the parameters of the encoder.
        It's using the Xavier initialization over Orthogonal initialization because it's more suitable for the ReLU activation function applied to the output of the attention layer.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:  # Check if it's a weight term
                nn.init.xavier_uniform_(param)  # Initialize weights using Xavier initialization
            elif 'bias' in name:  # Check if it's a bias term
                nn.init.constant_(param, 0)  # Initialize biases to zero

    def forward(self, state_t, context, mask):
        """
        This function computes the attention scores and returns the output.

        Args:
            state_t (torch.Tensor): The current state. (batch_size, 1, input_dim * 3 (GATembeding, first_node, end_node))
            context (torch.Tensor): The context to attend to. (batch_size, n_nodes, input_dim)
            mask (torch.Tensor): The mask to apply to the attention scores. (batch_size, n_nodes)

        Returns:
            out_put: The output of the attention layer.
        """

        # Compute Q
        Q = self.w(state_t).reshape(-1, 1)  # batch_size * hidden_dim
        
        # Compute the compatibility scores for calculating attention scores
        Q = self.norm * torch.matmul(Q, context.transpose(2,3))  # batch_size * k * 1 * hidden_dim * -1
        Q = Q / math.sqrt(self.head_dim)
        
        # Apply dropout to the attention scores
        Q = Q.masked_fill(mask == True, float("-inf"))  # Mask out the positions where mask is true

        # Compute the attention scores
        u_i = torch.matmul(Q, context.transpose(2,3))  # batch_size * n_nodes
        
        # Process the weighted sum of the context nodes, apply dropout and return the output
        out_put = torch.matmul(u_i, state_t).squeeze(1)  # batch_size * hidden_dim
        out_put = self.fc(out_put)

        return out_put
