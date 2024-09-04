import datetime
import pandas as pd
import torch
import torch.optim as optim
import os
import time
import dice_mc.torch as dice
from torch.utils.tensorboard import SummaryWriter

from ..RL.euclidean_cost import euclidean_cost
from ..train.utils import evaluate_on_validation

now = datetime.datetime.now().strftime("%Y-%m-%d %H")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

def train(model, data_loader, valid_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 2.0
    num_samples = 8
    
    # Instantiate the model and the optimizer
    actor = model.to(device)
    
    actor_optim = optim.Adam(actor.parameters(), lr)
    
    # Initialize an empty list to store results for pandas dataframe
    training_results = []
    train_start = time.time()
    
    best_validation_reward = float('inf')
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

            # Initialize lists to collect all costs and log probabilities
            surrogate_losses = []
            for sample_idx in range(num_samples):
                actions = actions_list[sample_idx]
                cost = euclidean_cost(batch.x, actions.detach(), batch)  # Shape: (batch_size,)
                
                surrogate_loss = dice.cost_node(cost, [log_ps_list[sample_idx]])
                surrogate_losses.append(surrogate_loss)
                baseline_term = dice.batch_baseline_term(cost, [log_ps_list[sample_idx]])
                surrogate_losses[sample_idx] = surrogate_loss + baseline_term       

            # Stack the costs and surrogate losses
            surrogate_loss_stack = torch.stack(surrogate_losses, dim=1)
            total_loss = surrogate_loss_stack.mean()  # Mean over the samples
            
            # Backward pass
            actor_optim.zero_grad()
            total_loss.backward()
            
            # Clip helps with the exploding and vanishing gradient problem
            total_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm, norm_type=2)
            
            # Update the actor
            actor_optim.step()

            # Update the pre-allocated tensors
            rewards[i] = torch.mean(cost.detach())
            losses[i] = torch.mean(total_loss.detach())
            
            # END OF EPOCH BLOCK CODE
        print(f'Epoch {epoch}, mean loss: {mean_loss:.2f}, mean reward: {mean_reward:.2f}, time: {epoch_time:.2f}')
        
        # Calculate the mean values for the epoch
        mean_reward = torch.mean(rewards).item()
        mean_loss = torch.mean(losses).item()
        # mean_parameters = torch.mean(parameters).item()

        # Push losses and rewards to tensorboard
        writer.add_scalar('Loss/Train', mean_loss, epoch)
        writer.add_scalar('Reward', mean_reward, epoch)
        # writer.add_scalar('Gradients/Total_Grad_Norm', mean_parameters, epoch)
        
    
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

    #     mean_validation_reward = evaluate_on_validation(actor, valid_loader, n_steps, T)
    #     print(f'Validation Reward: {mean_validation_reward:.3f}')
    
    # # Check if this is the best model so far
    #     if mean_validation_reward < best_validation_reward:
    #         best_validation_reward = mean_validation_reward
    #         print(f'New best model found with validation reward: {mean_validation_reward:.3f}')
        
    #         # Save the best model
    #         best_model_path = os.path.join(folder, 'best_actor.pt')
    #         torch.save(actor.state_dict(), best_model_path)
    #         print(f'Best model saved at {best_model_path}')

        # Save if the Loss is less than the minimum so far
        # epoch_dir = os.path.join(folder, '%s' % epoch)
        # if not os.path.exists(epoch_dir):
        #     os.makedirs(epoch_dir)
        # save_path = os.path.join(epoch_dir, 'actor.pt')
        # save_path_best = os.path.join(epoch_dir, 'actor_best.pt')
        # if mean_loss < min_loss_soFar:
        #     torch.save(actor.state_dict(), save_path_best)
        #     print(f'New best model saved at epoch {epoch}')
        #     min_loss_soFar = mean_loss
        # torch.save(actor.state_dict(), save_path)
        
        # Push losses and rewards to tensorboard
        writer.flush()

    training_end = time.time()
    training_time = training_end - train_start
    print(f' Total Training Time: {training_time:.2f}')
    writer.close()