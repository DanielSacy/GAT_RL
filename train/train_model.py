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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
now = datetime.datetime.now().strftime("%Y-%m-%d %H")

def train(model, data_loader, folder, filename, lr, n_steps, num_epochs, T):
    # Gradient clipping value
    max_grad_norm = 5.0
    
    # Instantiate the model and the optimizer
    actor = model.to(device)
    
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
        memory = torch.zeros(batch_size, device=device)

        epoch_start = time.time()
        
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            
            # Actor forward pass with multiple samples
            actions, log_p  = actor(batch, n_steps, greedy=False, T=T)

            # Compute the cost of the actions
            cost = euclidean_cost(batch.x, actions.detach(), batch)  # Shape: (batch_size,)
            
            # Compute the surrogate loss - DiCE
            surrogate_loss = dice.cost_node(cost, [log_p])
            baseline_term = dice.batch_baseline_term(cost, [log_p])
            surrogate_loss_baseline = surrogate_loss + baseline_term       
            total_loss = surrogate_loss_baseline.mean()  # Mean over the samples
            memory_allocated = torch.cuda.memory_allocated()

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
            memory[i] = memory_allocated /  (1024 ** 2)  # Convert to MB
            
            # END OF EPOCH BLOCK CODE
        
        # Calculate the mean values for the epoch
        mean_reward = torch.mean(rewards).item()
        mean_loss = torch.mean(losses).item()
        mean_memory = torch.mean(memory).item()

        # Time management
        end = time.time()
        epoch_time = end - epoch_start
        epoch_start = end
        elapsed_time = time.time() - train_start
        print(f'Epoch {epoch}, mean loss: {mean_loss:.2f}, mean reward: {mean_reward:.2f}, time: {epoch_time:.2f}, elapsed time: {elapsed_time:.2f}')
        
        # Push losses and rewards to tensorboard
        # writer.add_scalar('Loss/Train', mean_loss, epoch)
        # writer.add_scalar('Reward', mean_reward, epoch)
        writer.add_scalar('Loss/Train', mean_loss, elapsed_time)
        writer.add_scalar('Reward', mean_reward, elapsed_time)
        writer.add_scalar('Memory', mean_memory, elapsed_time)

        # Store the results for this epoch
        training_results.append({
            'epoch': epoch,
            'mean_reward': f'{mean_reward:.3f}',
            ' ': ' ',
            'mean_loss': f'{mean_loss:.3f}',
            ' ': ' ',
            'epoch_time': f'{epoch_time:.2f}',
            ' ': ' ',
            'memory': f'{mean_memory:.2f}'
        })

        # Convert the results to a pandas DataFrame
        results_df = pd.DataFrame(training_results)

        # Save the results to a CSV file
        results_df.to_csv(f'instances/{now}h.csv', index=False)

        # Save the model
        epoch_dir = os.path.join(folder, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)
        
        # Push losses and rewards to tensorboard
        writer.flush()

    training_end = time.time()
    training_time = training_end - train_start
    print(f' Total Training Time: {training_time:.2f}')
    writer.close()