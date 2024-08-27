import datetime
import pandas as pd
import torch
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
import dice_mc.torch as dice

from ..RL.Pairwise_cost import pairwise_cost

now = datetime.datetime.now().strftime("%Y-%m-%d %H")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

def train(model, data_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 2.0
    num_samples = 8
    
    # Instantiate the model and the optimizer
    actor = model.to(device)
    # baseline = RolloutBaseline(actor, valid_loader, n_nodes=n_steps, T=T)
    
    actor_optim = optim.Adam(actor.parameters(), lr)
    
    # Initialize an empty list to store results for pandas dataframe
    training_results = []
    train_start = time.time()
    
    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")
        actor.train()
        
        # Faster logging
        batch_size = len(data_loader)
        rewards = torch.zeros(batch_size, device=device)
        losses = torch.zeros(batch_size, device=device)

        times = []
        epoch_start = time.time()
        
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            # Actor forward pass with multiple samples
            actions_list, log_ps_list = actor(batch, n_steps, greedy=False, T=T, num_samples=num_samples)

            # Compute REWARD for each sample
            costs_list = []
            surrogate_losses = []
            for sample_idx in range(num_samples):
                actions = actions_list[sample_idx]
                cost = pairwise_cost(actions, batch)
                costs_list.append(cost)
                # Convert cost to reward
                # reward = -cost
                
                # Compute DiCE Surrogate Loss
                surrogate_loss = dice.cost_node(cost, [log_ps_list[sample_idx]])
                surrogate_losses.append(surrogate_loss)
                
                # Compute Baseline Term using REINFORCE with replacement for this sample
                baseline_term = dice.batch_baseline_term(surrogate_loss, [log_ps_list[sample_idx]])
                
                # Add the baseline term to the surrogate loss for this sample
                surrogate_losses[sample_idx] = surrogate_loss + baseline_term       

            # Stack the costs and surrogate losses
            surrogate_loss_stack = torch.stack(surrogate_losses, dim=1)
            # Combine surrogate loss and baseline term
            total_loss = surrogate_loss_stack.mean() # Mean over the samples
            
            # Actor Backward pass
            actor_optim.zero_grad()
            total_loss.backward()
            
            # Clip helps with the exploding and vanishing gradient problem
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm, norm_type=2)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm, norm_type=2)
            writer.add_scalar('Gradients/Total_Grad_Norm', total_grad_norm, epoch)
            
            # Update the actor
            actor_optim.step()

            # Update the pre-allocated tensors
            rewards[i] = torch.mean(cost.detach())
            losses[i] = total_loss.detach()
            
            # END OF EPOCH BLOCK CODE
        
        # Rollout baseline update
        # baseline.epoch_callback(actor, epoch)
          
        # Calculate the mean values for the epoch
        mean_reward = torch.mean(rewards).item()
        mean_loss = torch.mean(losses).item()

        # Push losses and rewards to tensorboard
        writer.add_scalar('Loss/Train', mean_loss, epoch)
        writer.add_scalar('Reward', mean_reward, epoch)
        
    
        # Print epoch Time
        end = time.time()
        epoch_time = end - epoch_start
        times.append(end - epoch_start)
        epoch_start = end

        # Store the results for this epoch
        training_results.append({
            'epoch': epoch,
            'mean_reward': f'{mean_reward:.3f}',
            # 'MIN_REWARD': f'{min_reward_soFar:.3f}',
            ' ': ' ',
            'mean_loss': f'{mean_loss:.3f}',
            # 'MIN_LOSS': f'{min_loss_soFar:.3f}',
            ' ': ' ',
            'epoch_time': f'{epoch_time:.2f}'
            # 'param_norm': f'{np.mean(param_norms):.3f}'
        })

        # Convert the results to a pandas DataFrame
        results_df = pd.DataFrame(training_results)

        # Save the results to a CSV file
        results_df.to_csv(f'instances/{now}h.csv', index=False)

        print(f'Epoch {epoch}, mean loss: {mean_loss:.3f}, mean reward: {mean_reward:.3f}, time: {epoch_time:.2f}')

        # Save if the Loss is less than the minimum so far
        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        # save_path_best = os.path.join(epoch_dir, 'actor_best.pt')
        # if mean_loss < min_loss_soFar:
        #     torch.save(actor.state_dict(), save_path_best)
        #     print(f'New best model saved at epoch {epoch}')
        #     min_loss_soFar = mean_loss
        torch.save(actor.state_dict(), save_path)
        
        # Push losses and rewards to tensorboard
        writer.flush()

    training_end = time.time()
    training_time = training_end - train_start
    print(f' Total Training Time: {training_time:.2f}')
    writer.close()