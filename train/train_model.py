import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time

from src_batch.RL.Compute_Reward import compute_reward
from src_batch.RL.MST_baseline_instance import mst_baseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter()

def adv_normalize(adv):
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train(model, data_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 10.0
    
    # Instantiate the model and the optimizer
    actor = model
    actor_optim = optim.Adam(actor.parameters(), lr)
    
    # Initialize an empty list to store results for pandas dataframe
    training_results = []
    min_reward_soFar = float('-inf')
    min_loss_soFar = float('inf')
    
    train_start = time.time()
    
    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")
        actor.train()

        times, losses, rewards, mst_rewards, param_norms = [], [], [], [], []
        epoch_start = time.time()
        start = epoch_start
        
        for batch in data_loader:
            batch = batch.to(device)
            
            # Actor forward pass
            actions, tour_logp = actor(batch, n_steps, greedy=False, T=T)
            # Append the depot {0} at the end of every route
            depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            actions = torch.cat([actions, depot_tensor.detach()], dim=1).detach()
            
            # Compute reward and baseline
            # Should detach the actions tensor to avoid backpropagating through the reward computation
            reward = compute_reward(actions.detach(), batch)
            mst_reward = mst_baseline(batch)
            
            # If the reward is less than the minimum so far, save it
            if torch.mean(reward.detach()).item() > min_reward_soFar:
                min_reward_soFar = torch.mean(reward).item()
                save_path = os.path.join(folder, 'best_actor.pt')
                torch.save(actor.state_dict(), save_path)

            # Compute advantage
            # Negative reward with negative baseline
            # Reward is greater than baseline so advantage is negative
            advantage = (reward - mst_reward)
                
            # Whiten advantage    
            # advantage_norm = adv_normalize(advantage)
            
            # Backward pass
            reinforce_loss = torch.mean(advantage.detach() * tour_logp)
            
            # If loss is less than the minimum so far, save it
            if reinforce_loss.item() < min_loss_soFar:
                min_loss_soFar = reinforce_loss.item()
                save_path = os.path.join(folder, 'best_actor.pt')
                torch.save(actor.state_dict(), save_path)
            
            actor_optim.zero_grad()
            reinforce_loss.backward()
            
            # Monitor gradients
            writer.add_scalar("Loss/train", reinforce_loss, epoch)
            for name, param in actor.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f'Gradients/{name}_grad_norm', param.grad.norm(), epoch)
                    
            # total_norm_before = 0.0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm_before += param_norm.item() ** 2
            # total_norm_before = total_norm_before ** 0.5
            # print(f"Total norm before clipping: {total_norm_before}")
            
            # Clip helps with the exploding and vanishing gradient problem
            # torch.nn.utils.clip_grad_value_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            # Step 3: Compute gradient norm after clipping
            # total_norm_after = 0.0
            # for p in actor.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm_after += param_norm.item() ** 2
            # total_norm_after = total_norm_after ** 0.5
            # print(f"Total norm after clipping: {total_norm_after}")
            # for name, p in model.named_parameters():
            #     if p.grad is not None:
            #         # print(f'{name} grad norm before: {p.grad.data.norm(2)}')
            #         torch.nn.utils.clip_grad_norm_(p, max_grad_norm)
            #         # print(f'{name} grad norm after: {p.grad.data.norm(2)}')

            
            # Update the actor
            actor_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            mst_rewards.append(torch.mean(mst_reward.detach()).item())
            losses.append(torch.mean(reinforce_loss.detach()).item())
            # param_norms.append(total_norm_before).detach()

        # Print epoch Time
        end = time.time()
        epoch_time = end - epoch_start
        times.append(end - epoch_start)
        epoch_start = end

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_mst_reward = np.mean(mst_rewards)
        
        # Store the results for this epoch
        training_results.append({
            'epoch': epoch,
            'mean_reward': f'{mean_reward:.3f}',
            'MIN_REWARD': f'{min_reward_soFar:.3f}',
            'mean_mst_reward': f'{mean_mst_reward:.3f}',
            ' ': ' ',
            'mean_loss': f'{mean_loss:.3f}',
            'MIN_LOSS': f'{min_loss_soFar:.3f}',
            ' ': ' ',
            'epoch_time': f'{epoch_time:.2f}'
            # 'param_norm': f'{np.mean(param_norms):.3f}'
        })

        # Convert the results to a pandas DataFrame
        results_df = pd.DataFrame(training_results)

        # Save the results to a CSV file
        results_df.to_csv('instances/Training_Results.csv', index=False)

        print(f'Epoch {epoch}, mean loss: {mean_loss:4f}, mean reward: {mean_reward}, time: {epoch_time:.2f}')

        # Save the model if it is the best so far
        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)
        
        # Push losses and rewards to tensorboard
        writer.flush()

    training_end = time.time()
    training_time = training_end - train_start
    print(f'epoch {epoch}, mean loss: {np.mean(losses)}, mean reward: {np.mean(rewards)}, time: {training_time:.2f}')
    writer.close()