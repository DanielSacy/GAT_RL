import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerAttention(nn.Module):
    """
    This class computes the attention scores and returns the output.
    Args:
    - n_heads: The number of heads.
    - cat: The number of features to concatenate.
    - input_dim: The dimension of the input.
    - hidden_dim: The dimension of the hidden layer.
    - attn_dropout: The dropout rate for the attention scores.
    - dropout: The dropout rate for the output.
    
    Returns:
    - out_put: The output of the attention layer.
    """
    
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(TransformerAttention, self).__init__()
        
        "Assert that the hidden dimension is divisible by the number of heads."
        if hidden_dim % n_heads != 0:
            raise ValueError(f'hidden_dim({hidden_dim}) should be divisible by n_heads({n_heads}).')

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)
        
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

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

    def forward(self, state_t, context, mask):
        """
        This function computes the attention scores and returns the output.
        
        Args:
        - state_t: The current state. (batch_size, 1, input_dim * 3 (GATembeding, first_node, end_node))
        - context: The context to attend to. (batch_size, n_nodes, input_dim)
        - mask: The mask to apply to the attention scores. (batch_size, n_nodes)
        
        Returns:
        - out_put: The output of the attention layer. (batch_size, hidden_dim)
        """
        
        # Compute Q, K, and V
        batch_size, n_nodes, input_dim = context.size()
        # Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        # K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        # V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q = self.w(state_t).reshape(batch_size, 1, self.n_heads, -1)
        K = self.k(context).reshape(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).reshape(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Compute compatibility scores for calculating attention scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(2,3)) # (batch_size, n_heads, 1, hidden_dim) * (batch_size, n_heads, hidden_dim, n_nodes)
        
        # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        compatibility = compatibility.squeeze(2)  # (batch_size, n_heads, n_nodes)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        # Compute attention scores
        scores = F.softmax(u_i, dim=-1)  # (batch_size,n_heads,n_nodes)
        scores = scores.unsqueeze(2)
        
        # Process the weighted sum of the context nodes, apply dropout and return the output
        out_put = torch.matmul(scores, V)  
        # out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)
        out_put = out_put.squeeze(2).reshape(batch_size, self.hidden_dim)
        out_put = self.fc(out_put)

        return out_put  # out_put: (batch_size, hidden_dim)