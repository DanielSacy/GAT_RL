import datetime
import pandas as pd
import torch
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from ..RL.euclidean_cost import euclidean_cost
from ..RL.Rollout_Baseline import RolloutBaseline, rollout

now = datetime.datetime.now().strftime("%Y-%m-%d %H")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

def train(model, data_loader, valid_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 2.0
    
    # Instantiate the model and the optimizer
    actor = model.to(device)
    baseline = RolloutBaseline(actor, valid_loader, n_nodes=n_steps, T=T)
    
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
        baselines = torch.zeros(batch_size, device=device)
        advantages = torch.zeros(batch_size, device=device)
        losses = torch.zeros(batch_size, device=device)
        parameters = torch.zeros(batch_size, device=device)

        times = []
        epoch_start = time.time()
        
        # scheduler = LambdaLR(actor_optim, lr_lambda=lambda f: 0.96 ** epoch)
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            
            # Actor forward pass
            actions, tour_logp = actor(batch, n_steps, greedy=False, T=T)
            
            # REWARD
            # Track or not the gradients of the reward
            # reward = pairwise_cost(actions, batch)
            cost = euclidean_cost(batch.x, actions.detach(), batch)
            
            '''BASELINE BLOCK'''        
            # EXPONENTIAL MOVING AVERAGE
            # Put it in inference mode?
            # if i == 0:
            #     critic_exp_mvg_avg = cost.mean()
            # else:
            #     critic_exp_mvg_avg = (critic_exp_mvg_avg * 0.8) + ((1. - 0.8) * cost.mean()).detach()
            
            # ROLLOUT
            rollout_cost = baseline.eval(batch, n_steps)
            '''BASELINE BLOCK'''            
            
            # ADVANTAGE
            advantage = (cost - rollout_cost)
            
            # Normalize the advantage
            # advantage_norm = normalize(advantage)
            # print("advantage_norm:", advantage_norm)
            
            # Actor Backward pass
            reinforce_loss = torch.mean(advantage.detach() * tour_logp)
            actor_optim.zero_grad()
            reinforce_loss.backward()
            
            # Clip helps with the exploding and vanishing gradient problem
            total_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm, norm_type=2)
            
            # Update the actor
            actor_optim.step()
            # scheduler.step()

            # Update the pre-allocated tensors
            rewards[i] = torch.mean(cost.detach())
            baselines[i] = torch.mean(rollout_cost.detach())
            advantages[i] = torch.mean(advantage.detach())
            losses[i] = torch.mean(reinforce_loss.detach())
            parameters[i] = total_grad_norm.detach()
            
            # END OF EPOCH BLOCK CODE
        
        # Rollout baseline update
        baseline.epoch_callback(actor, epoch)
          
        # Calculate the mean values for the epoch
        mean_reward = torch.mean(rewards).item()
        mean_baseline = torch.mean(baselines).item()
        mean_advantage = torch.mean(advantages).item()
        mean_loss = torch.mean(losses).item()
        mean_parameters = torch.mean(parameters).item()

        # Push losses and rewards to tensorboard
        writer.add_scalar('Loss/Train', mean_loss, epoch)
        writer.add_scalar('Reward', mean_reward, epoch)
        writer.add_scalar('Gradients/Total_Grad_Norm', mean_parameters, epoch)
        
    
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
            'mean_baseline': f'{mean_baseline:.3f}',
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
        results_df.to_csv(f'instances/{now}h.csv', index=False)

        # print(f'Epoch {epoch}, mean loss: {mean_loss:.2f}, mean reward: {mean_reward:.2f}, time: {epoch_time:.2f}')
        print(f'Epoch {epoch}, mean loss: {mean_loss:.3f}, mean reward: {mean_reward:.3f}, mean_baseline: {mean_baseline:.3f}, mean_advantage: {mean_advantage:.3f}, time: {epoch_time:.2f}')

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