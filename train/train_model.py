from copy import deepcopy
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time

from src_batch.RL.Compute_Reward import compute_reward
from src_batch.RL.Pairwise_reward import pairwise_reward
from src_batch.RL.MST_baseline_instance import mst_baseline
from src_batch.RL.MovingAvg_baseline import MovingAverageBaseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter()

def adv_normalize(adv):
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train(model, data_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 5.0
    
    # Instantiate the model and the optimizer
    actor = model.to(device)
    baseline = deepcopy(model).to(device)
    # moving_avg_baseline = MovingAverageBaseline(beta=0.9)
    
    actor_optim = optim.Adam(actor.parameters(), lr)
    
    # Initialize an empty list to store results for pandas dataframe
    training_results = []
    # min_reward_soFar = float('-inf')
    # min_loss_soFar = float('inf')
    
    train_start = time.time()
    
    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")
        actor.train()
        baseline.eval()
        
        # Faster logging
        batch_size = len(data_loader)
        rewards = torch.zeros(batch_size, device=device)
        BL_rewards = torch.zeros(batch_size, device=device)
        advantages = torch.zeros(batch_size, device=device)
        losses = torch.zeros(batch_size, device=device)

        times = []
        # times, losses, rewards, BL_rewards, advantages, param_norms = [], [], [], [], [], []
        
        epoch_start = time.time()
        for i, batch in enumerate(data_loader):
            # Sent data to device when generating the data
            batch = batch.to(device)
            
            # Actor forward pass
            actions, tour_logp = actor(batch, n_steps, greedy=False, T=T)
            # Append the depot {0} at the end of every route
            # depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            # actions = torch.cat([actions, depot_tensor], dim=1)
            
            # # Baseline forward pass
            with torch.inference_mode():
                BL_actions, _ = baseline(batch, n_steps, greedy=True, T=T)
                # BL_actions = torch.cat([BL_actions, depot_tensor], dim=1)
            
            # Compute reward and baseline
            reward = pairwise_reward(actions.detach(), batch)
            BL_reward = pairwise_reward(BL_actions.detach(), batch)
            # avg_baseline = moving_avg_baseline.update(reward)
            # mst_reward = mst_baseline(batch)

            # Compute advantage
            advantage = (reward - BL_reward)
                
            # Whiten advantage    
            # advantage_norm = adv_normalize(advantage)
            
            # Backward pass
            reinforce_loss = torch.mean(advantage.detach() * tour_logp)
            actor_optim.zero_grad()
            reinforce_loss.backward()
            
            # Clip helps with the exploding and vanishing gradient problem
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm, norm_type=2)
                     
            # Update the actor
            actor_optim.step()

            # Update the pre-allocated tensors
            rewards[i] = torch.mean(reward)
            BL_rewards[i] = torch.mean(BL_reward)
            advantages[i] = torch.mean(advantage)
            losses[i] = torch.mean(reinforce_loss)
            # Update lists
            # rewards.append(torch.mean(reward))
            # BL_rewards.append(torch.mean(BL_reward))
            # advantages.append(torch.mean(advantage))
            # losses.append(torch.mean(reinforce_loss))
            
        # Calculate the mean values for the epoch
        mean_reward = torch.mean(rewards).item()
        mean_BL_rewards = torch.mean(BL_rewards).item()
        mean_advantage = torch.mean(advantages).item()
        mean_loss = torch.mean(losses).item()

        # Push losses and rewards to tensorboard
        # if epoch % 30 == 0:
        #     baseline = deepcopy(actor)
        if epoch % 10 == 0:
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
            'mean_BL_rewards': f'{mean_BL_rewards:.3f}',
            'mean_advantage': f'{mean_advantage:.3f}',
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
        results_df.to_csv('instances/model_checkpoints__.csv', index=False)

        print(f'Epoch {epoch}, mean loss: {mean_loss:4f}, mean reward: {mean_reward}, mean_advantage: {mean_advantage}, time: {epoch_time:.2f}')

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