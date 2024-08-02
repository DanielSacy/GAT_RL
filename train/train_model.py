from ast import Param
import torch
import torch.optim as optim
import numpy as np
import os
import time
import logging
from src_batch.RL.Compute_Reward import compute_reward

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")


def adv_normalize(adv):
    # std = adv.std()
    # assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train(model, rol_baseline, data_loader, validation_loader, folder, filename, lr, n_steps, num_epochs, T):
    # max_grad_norm = 1.0
    
    actor = model
    actor.train()
    rollout = rol_baseline
    
    actor_optim = optim.Adam(actor.parameters(), lr)

    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")

        times, losses, rewards = [], [], []
        epoch_start = time.time()
        start = epoch_start
        for batch in data_loader:
            batch = batch.to(device)
            
            # Actor forward pass
            actions, tour_logp, depot_visits = actor(batch, n_steps, greedy=False, T=T)
            
            # Compute reward and baseline
            reward = compute_reward(actions, batch)
            bl_reward = rollout.rollout(batch, n_steps)
                     
            # Compute advantage
            advantage = (reward.detach() - bl_reward.detach())
            if not advantage.ne(0).any():
                print("advantage==0.")
            
            # Whiten advantage    
            advantage = adv_normalize(advantage)
            print("advantage normalized:", advantage)
            
            # Com Arvore Geradora Minima a vantagem será sempre negativa
            reinforce_loss = (advantage.detach() * tour_logp).mean()
            # reinforce_loss = torch.mean(advantage.detach() * tour_logp)
            
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

        print(f'Epoch {epoch}, mean loss: {mean_loss}, mean reward: {mean_reward}, time: {times}')

        # rol_baseline.epoch_callback(actor, epoch)

        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)
        
        
    logging.debug(f'epoch {epoch}, mean loss: {np.mean(losses)}, mean reward: {np.mean(rewards)}')