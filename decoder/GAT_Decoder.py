import logging
import torch
from torch import nn
from torch.distributions import Categorical
from dice_mc.torch import sample_categorical
from src_batch.decoder.PointerAttention import PointerAttention
from src_batch.decoder.mask_capacity import update_state, update_mask

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
class GAT_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GAT_Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.pointer = PointerAttention(8, input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim+1, hidden_dim, bias=False) # +1 to adjust for the concatenated capacity
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
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
        
    def forward(self, encoder_inputs, pool, capacity, demand, n_steps, T, greedy, num_samples):
        # encoder_inputs: (batch_size, n_nodes, hidden_dim)
        device = encoder_inputs.device  # to ensure the tensors are on the same device

        batch_size = encoder_inputs.size(0)  # Indicates the number of graphs in the batch
        seq_len = encoder_inputs.size(1)  # Feature dimension

        log_ps_list = []  # To hold multiple sampled log-probs for each sample
        actions_list = []  # To hold multiple sampled actions for each sample
        
        for _ in range(num_samples):
            mask1 = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
            mask = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
            dynamic_capacity = capacity.expand(batch_size, -1).to(device)
            demands = demand.to(device)
            index = torch.zeros(batch_size, dtype=torch.long, device=device)
            log_ps = []
            actions = []    # encoder_inputs: (batch_size, n_nodes, hidden_dim)

            # finished_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)  # Track finished samples

            # i=0
            # while (mask1[:, 1:].sum(1) < (demand.size(1) - 1)).any():
            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:   
                    _input = encoder_inputs[:, 0, :]  # depot (batch_size,node,hidden_dim)
                    # print(f'_input: {_input}\n\n')

                # pool+cat(first_node,current_node)
                decoder_input = torch.cat([_input, dynamic_capacity], -1)
                decoder_input = self.fc(decoder_input)
                pool = self.fc1(pool.to(device))
                decoder_input = decoder_input + pool

                # If it is the first step, update the mask to avoid visiting the depot again
                if i == 0:
                    mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)

                # Compute the probability distribution         
                p = self.pointer(decoder_input, encoder_inputs, mask,T)
                dist = Categorical(p)
                
                if greedy:
                    _, index = p.max(dim=-1)
                else:
                    # log_p, index = sample_categorical(p)
                    index = dist.sample()

                actions.append(index.data.unsqueeze(1))
                
                # Update the dynamic capacity and mask
                dynamic_capacity = update_state(demands, dynamic_capacity, index.unsqueeze(-1), capacity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)
                
                # Apply mask to the log probabilities
                log_p = dist.log_prob(index)
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1))).float()
                # is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                log_p = log_p * (1. - is_done)
                log_ps.append(log_p.unsqueeze(1))
                
                # Update finished_samples tensor
                # finished_samples = finished_samples | (mask1[:, 1:].eq(0).all(dim=-1))

                _input = torch.gather(
                                      encoder_inputs, 1,
                                      index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,encoder_inputs.size(2))
                                      ).squeeze(1)

            # Concatenate the actions and log probabilities
            log_ps = torch.cat(log_ps, dim=1)
            actions = torch.cat(actions, dim=1)
            
            log_p = log_ps.sum(dim=1) # Dimension of log_p: (batch_size,)
            
            actions_list.append(actions)
            log_ps_list.append(log_p)

        return actions_list, log_ps_list