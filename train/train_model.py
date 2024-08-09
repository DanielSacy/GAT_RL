from ast import Param
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import os
import time
import logging

from src_batch.RL.Compute_Reward import compute_reward
from src_batch.RL.MST_baseline_instance import mst_baseline

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")


def adv_normalize(adv):
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train(model, rol_baseline, data_loader, validation_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 5.0
    
    # Instantiate the model and the optimizer
    actor = model
    actor.train()
    actor_optim = optim.Adam(actor.parameters(), lr)
    
    # Best model parameters so far
    best_model = None
    
    # Initialize an empty list to store results for pandas dataframe
    training_results = []

    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")

        times, losses, rewards, mst_rewards = [], [], [], []
        epoch_start = time.time()
        start = epoch_start
        for batch in data_loader:
            batch = batch.to(device)
            
            # Actor forward pass
            actions, tour_logp = actor(batch, n_steps, greedy=False, T=T)
            # Append the depot {0} at the end of every route
            depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            actions = torch.cat([actions, depot_tensor], dim=1).detach()
            
            # Compute reward and baseline
            # Should detach the actions tensor to avoid backpropagating through the reward computation
            reward = compute_reward(actions.detach(), batch)
            mst_reward = mst_baseline(batch)
             
            # Compute advantage
            # Negative reward with negative baseline
            # Reward is greater than baseline so advantage is negative
            advantage = (reward - mst_reward)
                
            # Whiten advantage    
            # advantage_norm = adv_normalize(advantage)
            
            # Compute the reinforce loss
            reinforce_loss = (advantage * tour_logp).mean()
            
            # Backward pass
            actor_optim.zero_grad()
            reinforce_loss.backward()
            
            # Clip helps with the exploding and vanishing gradient problem
            torch.nn.utils.clip_grad_value_(actor.parameters(), max_grad_norm)
            
            # Update the actor
            actor_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            mst_rewards.append(torch.mean(mst_reward.detach()).item())
            losses.append(torch.mean(reinforce_loss.detach()).item())

        # Print epoch Time
        end = time.time()
        epoch_time = end - start
        times.append(end - start)
        start = end

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_mst_reward = np.mean(mst_rewards)
        # Store the results for this epoch
        training_results.append({
            'epoch': epoch,
            'mean_reward': f'{mean_reward:.3f}',
            'mean_mst_reward': f'{mean_mst_reward:.3f}',
            'mean_loss': f'{mean_loss:.3f}',
            'epoch_time': f'{epoch_time:.3f}'
        })

        # Convert the results to a pandas DataFrame
        results_df = pd.DataFrame(training_results)

        # Save the results to a CSV file
        results_df.to_csv('instances/Training_Results.csv', index=False)

        logging.debug(f'Epoch {epoch}, mean loss: {mean_loss:4f}, mean reward: {mean_reward}, time: {epoch_time:.2f}')

        # Save the model if it is the best so far
        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        if best_model is None or mean_reward < np.mean(rewards):
            best_model = torch.save(actor.state_dict(), save_path)
            print(f'Saved model at epoch {epoch} with mean reward: {mean_reward}')
        # torch.save(actor.state_dict(), save_path)
         
    logging.debug(f'epoch {epoch}, mean loss: {np.mean(losses)}, mean reward: {np.mean(rewards)}')