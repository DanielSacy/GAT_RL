from ast import Param
import torch
import torch.optim as optim
import numpy as np
import os
import time
import logging

from ..RL.Compute_Reward import compute_reward
from ..RL.MST_baseline_instance import mst_baseline
#test
from ..RL.MST_baseline_dynamic import mst_baseline as mst_baseline_dynamic

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")


def adv_normalize(adv):
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train(model, rol_baseline, data_loader, validation_loader, folder, filename, lr, n_steps, num_epochs, T):
    # max_grad_norm = 1.0
    
    actor = model
    actor.train()
    # rollout = rol_baseline
    
    actor_optim = optim.Adam(actor.parameters(), lr)

    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")

        times, losses, rewards = [], [], []
        epoch_start = time.time()
        start = epoch_start
        for batch in data_loader:
            batch = batch.to(device)
            
            # Actor forward pass
            actions, tour_logp = actor(batch, n_steps, greedy=False, T=T)
            # Append the depot {0} at the end of every route
            depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            actions = torch.cat([actions, depot_tensor], dim=1)
            
            # Compute reward and baseline
            reward = compute_reward(actions, batch)
            # bl_reward = rollout.rollout(batch, n_steps)
            mst_reward = mst_baseline(batch)
            
            # mst_dynamic_reward = mst_baseline_dynamic(batch)
            # print(f'mst_dynamic_reward: {mst_dynamic_reward}')
                     
            # Compute advantage
            advantage = (reward - mst_reward)
            # advantage = (reward.detach() - bl_reward.detach())
            if not advantage.ne(0).any():
                print("advantage==0.")
                
            # Whiten advantage    
            advantage = adv_normalize(advantage)
            # print(f'advantage normalized: {advantage}\n')
            
            # Com Arvore Geradora Minima a vantagem será sempre negativa - SÓ QUE NÃO
            reinforce_loss = -(advantage.detach() * tour_logp).mean()
            
            # Backward pass
            actor_optim.zero_grad()
            reinforce_loss.backward()
            # torch.nn.utils.clip_grad_value_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(reinforce_loss.detach()).item())

        # Print epoch statistics
        end = time.time()
        times.append(end - start)
        start = end

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        # mean_loss = np.mean(losses[epoch])
        # mean_reward = np.mean(rewards[epoch])

        logging.debug(f'Epoch {epoch}, mean loss: {mean_loss}, mean reward: {mean_reward}, time: {times}')

        # rol_baseline.epoch_callback(actor, epoch)

        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)
        
        
    logging.debug(f'epoch {epoch}, mean loss: {np.mean(losses)}, mean reward: {np.mean(rewards)}')