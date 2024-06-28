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
        # if INIT:
        #     for name, p in self.named_parameters():
        #         if 'weight' in name:
        #             if len(p.size()) >= 2:
        #                 nn.init.orthogonal_(p, gain=1)
        #         elif 'bias' in name:
        #             nn.init.constant_(p, 0)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        This function initializes the parameters of the attention layer.
        It's using the Xavier initialization over Orthogonal initialization because it's more suitable for the ReLU activation function applied to the output of the attention layer.
        
        SHOULD I PUT THE IF INIT HERE?
        """
        nn.init.xavier_uniform_(self.w.weight.data)
        nn.init.xavier_uniform_(self.k.weight.data)
        nn.init.xavier_uniform_(self.v.weight.data)
        nn.init.xavier_uniform_(self.fc.weight.data)

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
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Compute compatibility scores for calculating attention scores
        compatibility = self.norm * torch.matmul(Q, K.transpose(2,3))  # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        compatibility = compatibility.squeeze(2)  # (batch_size,n_heads,n_nodes)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        # Compute attention scores and apply dropout to better generalize
        scores = F.softmax(u_i, dim=-1)  # (batch_size,n_heads,n_nodes)
        scores = F.dropout(scores, p=self.attn_dropout, training=self.training)
        scores = scores.unsqueeze(2)
        
        # Process the weighted sum of the context nodes, apply dropout and return the output
        out_put = torch.matmul(scores, V)  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)  # （batch_size,n_heads,hidden_dim）
        out_put = F.dropout(out_put, p=self.dropout, training=self.training) # Avoid overfitting
        out_put = self.fc(out_put)

        return out_put  # (batch_size,hidden_dim)