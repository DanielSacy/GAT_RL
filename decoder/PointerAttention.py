import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from decoder.TransformerAttention import TransformerAttention

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
        """
        nn.init.xavier_uniform_(self.k.weight.data)


    def forward(self, state_t, context, mask,T):
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
        print(f'batch_size: {batch_size}, n_nodes: {n_nodes}, input_dim: {input_dim}')
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # (batch_size,1,n_nodes)
        print(f'compatibility: {compatibility}')
        compatibility = compatibility.squeeze(1)
        print(f'compatibility 2: {compatibility}')
        x = torch.tanh(compatibility)
        print(f'x TANH: {x}')
        x = x * (10)
        print(f'x1: {x}')
        # min_value = torch.finfo(x.dtype).min
        min_value = 1e-15
        print(f'min_value type: {type(min_value)}')
        # x = x.masked_fill(mask.bool(), min_value)
        x = x.masked_fill(mask.bool(), float("-inf"))
        print(f'x2: {x}')
        
        # Compute the softmax scores
        scores = F.softmax(x / T, dim=-1)
        print(f'scores: {scores}')
        return scores