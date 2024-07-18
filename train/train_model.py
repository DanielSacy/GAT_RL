import torch
import torch.optim as optim
import numpy as np
import os
import time
from src_batch.RL.Compute_Reward import compute_reward
from src_batch.RL.Rollout_Baseline import rollout

def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train(model, rol_baseline, data_loader, validation_loader, folder, lr, n_steps, num_epochs, T):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_grad_norm = 2.0
    
    actor = model #Model(3, 1, hidden_node_dim, hidden_edge_dim, 0.0, conv_layers, 8, torch.tensor([steps])).to(device)
    rol_baseline = rol_baseline #RolloutBaseline(actor, validation_loader, n_nodes=steps)
    
    actor_optim = optim.Adam(actor.parameters(), lr=lr)

    costs = []
    for epoch in range(num_epochs):
        print("epoch:", epoch, "------------------------------------------------")
        actor.train()

        times, losses, rewards = [], [], []
        epoch_start = time.time()
        start = epoch_start

        for batch_idx, data in enumerate(data_loader):
            batch = batch.to(device)
            tour_indices, tour_logp, depot_visits = actor(data, n_steps, greedy=False, T=T)

            reward = compute_reward(tour_indices.detach(), data)
            base_reward = rol_baseline.eval(data, n_steps)

            advantage = (reward - base_reward)
            if not advantage.ne(0).any():
                print("advantage==0.")
                
            # Normalize the advantage
            advantage = adv_normalize(advantage)
            actor_loss = torch.mean(advantage.detach() * tour_logp)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            step = 200
            if (batch_idx + 1) % step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-step:])
                mean_reward = np.mean(rewards[-step:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_reward, mean_loss, times[-1]))

        rol_baseline.epoch_callback(actor, epoch)

        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)
        
        cost = rollout(actor, validation_loader, batch_size=len(validation_loader), n_nodes=steps)
        cost = cost.mean()
        costs.append(cost.item())

        print('Problem:TSP''%s' % steps, '/ Average distance:', cost.item())
        print(costs)
