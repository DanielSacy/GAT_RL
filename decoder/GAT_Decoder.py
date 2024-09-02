import logging
import torch
from torch import nn
from torch.distributions import Categorical
from decoder.PointerAttention import PointerAttention
from decoder.mask_capacity import update_state, update_mask

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
# Define the GAT_Decoder class
class GAT_Decoder(nn.Module):
    """
    A decoder module for the Graph Attention Network (GAT) model.
    
    Attributes:
        input_dim: The dimension of the input data.
        hidden_dim: The dimension of the hidden state.
        
        pointer: An instance of the PointerAttention class, used to compute attention weights.
        fc: A fully connected layer with ReLU activation, used for feature transformation.
        fc1: Another fully connected layer with ReLU activation, used for feature transformation and concatenation.
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(GAT_Decoder, self).__init__()
        
        # Initialize the input dimension and hidden state dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize the pointer attention module
        self.pointer = PointerAttention(8, input_dim, hidden_dim)
        
        # Initialize the fully connected layers for feature transformation
        self.fc = nn.Linear(hidden_dim+1, hidden_dim, bias=False)  # +1 to adjust for the concatenated capacity
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
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
        
    def forward(self, encoder_inputs, pool, capacity, demand, n_steps, T, greedy):
        """
        This function implements the forward pass of the decoder module.
        
        Parameters:
            encoder_inputs: The input data from the encoder (batch_size, n_nodes, hidden_dim)
            pool: The pooling result from the encoder (batch_size, hidden_dim)
            capacity: The maximum capacity for each node (batch_size,)
            demand: The demand for each node (demand.size(1) - 1)
            n_steps: The number of steps to perform in the decoder
            T: A tensor with shape (batch_size,), representing some unknown information
            greedy: A boolean flag indicating whether to use greedy or sampling-based decoding
        
        Returns:
            actions: The sequence of actions taken by the decoder (n_steps, batch_size)
            log_p: The log probability distribution over the actions (batch_size,)
        """
        
        # Get the device where the tensors are located
        device = encoder_inputs.device
        
        # Initialize the mask and index tensor to keep track of visited nodes
        batch_size = encoder_inputs.size(0)  # Indicates the number of graphs in the batch
        seq_len = encoder_inputs.size(1)    # Feature dimension
        
        # Initialize the mask and index tensors
        mask1 = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
        mask = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
        
        # Initialize the dynamic capacity and demands
        dynamic_capacity = capacity.expand(batch_size, -1).to(device)
        demands = demand.to(device)

        # Initialize the index tensor to keep track of the visited nodes
        index = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Initialize the log probabilities and actions tensors
        log_ps = []
        actions = []

        # Iterate over each step in the decoder
        for i in range(n_steps):
            if not mask1[:, 1:].eq(0).any():
                break
            
            # If it's the first step, update the mask to avoid visiting the depot again
            if i == 0:   
                _input = encoder_inputs[:, 0, :]  # depot (batch_size,node,hidden_dim)
                
                # Compute the attention weights and probability distribution
                decoder_input = torch.cat([_input, dynamic_capacity], -1)
                decoder_input = self.fc(decoder_input)
                pool = self.fc1(pool.to(device))
                decoder_input = decoder_input + pool
            mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)
            
            # Compute the attention weights and probability distribution
            p = self.pointer(decoder_input, encoder_inputs, mask,T)
                
            # Calculate the probability distribution for sampling
            dist = Categorical(p)
            
            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()
            
            # Update the actions and log probabilities with the current action
            actions.append(index.data.unsqueeze(1))
            log_p = dist.log_prob(index)
            is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
            log_p = log_p * (1. - is_done)

            # Update the log probabilities by adding the current log probability to the cumulative sum
            log_ps.append(log_p.unsqueeze(1))

            # Update the dynamic capacity and demands based on the current action
            dynamic_capacity = update_state(demands, dynamic_capacity, index.unsqueeze(-1), capacity[0].item())
            mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)
            
            # Get the input features for the next step based on the current action
            _input = torch.gather(
                                  encoder_inputs, 1,
                                  index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,encoder_inputs.size(2))
                                  ).squeeze(1)
            
        # Concatenate the actions and log probabilities
        log_ps = torch.cat(log_ps, dim=1)
        actions = torch.cat(actions, dim=1)
        log_p = log_ps.sum(dim=1) # Dimension of log_p: (batch_size,)

        return actions, log_p