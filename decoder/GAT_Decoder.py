import logging
import torch
from torch import nn
from torch.distributions import Categorical
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

        self.fc = nn.Linear(hidden_dim+1, hidden_dim, bias=False) # +1 to adjust for the concatenated capacity in line 52
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.reset_parameters()
        
    def reset_parameters(self):
        """
        This function initializes the parameters of the attention layer.
        It's using the Xavier initialization over Orthogonal initialization because it's more suitable for the ReLU activation function applied to the output of the attention layer.
        """
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.xavier_uniform_(self.fc1.weight.data)

    def forward(self, encoder_inputs, pool, capacity, demand, n_steps,T, greedy, depot_visits):
        
        device = encoder_inputs.device

        batch_size = encoder_inputs.size(0)  # Indicates the number of graphs in the batch
        seq_len = encoder_inputs.size(1)    # Feature dimension
        
        # n_steps = seq_len +1 # To account for the route starting and ending at the depot

        mask1 = encoder_inputs.new_zeros(batch_size, seq_len, device=device)
        mask = encoder_inputs.new_zeros(batch_size, seq_len, device=device)

        dynamic_capacity = capacity.expand(batch_size, -1).to(device)
        demands = demand.to(device)

        index = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Debugging log
        # logging.debug(f'encoder_inputs: {encoder_inputs.shape}')
        # logging.debug(f'batch_size: {batch_size}')
        # logging.debug(f'seq_len: {seq_len}')
        # logging.debug(f'dynamic_capacity: {dynamic_capacity}')
        # logging.debug(f'demands: {demands.shape}')

        log_ps = []
        actions = []
        count = 0
        i=0
        while (mask1[:, 1:].sum(1) < (demand.size(1) - 1)).any():
        # for i in range(n_steps):
        #     if not mask1[:, 1:].eq(0).any():
        #         print(f'Breaking at i={i}, mask1: {mask1}')
        #         break
            if i == 0:   
                _input = encoder_inputs[:, 0, :]  # depot (batch_size,node,hidden_dim)
                # print(f'_input: {_input}\n\n')

            # pool+cat(first_node,current_node)
            decoder_input = torch.cat([_input, dynamic_capacity], -1)
            decoder_input = self.fc(decoder_input)
            pool = self.fc1(pool.to(device))
            decoder_input = decoder_input + pool
            
            # If it's the first step, update the mask
            if i == 0:
                mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)
            
            
            p = self.pointer(decoder_input, encoder_inputs, mask,T)
            
            # # Check for NaN values in p
            # if torch.isnan(p).any():
            # #     # logging.warning("NaN values found in p. Replacing with zeroes.")
            #     p = torch.nan_to_num(p, 0.1)

            # Calculate the probability distribution for sampling
            try:
                dist = Categorical(p)
            except RuntimeError as e:
                e = str(e)
            finally:
                print(f'\n\ndecoder_input: {decoder_input}')
                print(f'encoder_inputs: {encoder_inputs}')
                print(f'mask: {mask}')
                print(f'T: {T}\n\n')
                
            # dist = Categorical(p)
            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()
                
            actions.append(index.data.unsqueeze(1))
            log_p = dist.log_prob(index)
            is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
            # logging.debug(f'is_done: {is_done}')
            log_p = log_p * (1. - is_done)

            log_ps.append(log_p.unsqueeze(1))

            dynamic_capacity, depot_visits = update_state(demands, dynamic_capacity, index.unsqueeze(-1), capacity[0].item(), depot_visits)
            mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)
            
            _input = torch.gather(
                                  encoder_inputs, 1,
                                  index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,encoder_inputs.size(2))
                                  ).squeeze(1)
            
            i+=1
            # count += 1
            # if count % 1000 == 0:
            #     logging.debug(f'count: {count}')
        log_ps = torch.cat(log_ps, dim=1)
        # logging.debug(f'actions: {actions}')
        actions = torch.cat(actions, dim=1)

        log_p = log_ps.sum(dim=1)

        return actions, log_p, depot_visits