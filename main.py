import pandas as pd
import torch
import os
import logging
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model

from RL.Compute_Reward import compute_reward

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# CUDA is used if available, otherwise CPU is used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def prepare_data(data_path, batch_size=1):
    IG = InstanceGenerator()
    data_loader = IG.get_dataloader(data_path, batch_size=batch_size)
    return data_loader

def load_model(node_input_dim, edge_input_dim, hidden_dim, layers, negative_slope, dropout, model_path, device):
    model = Model(node_input_dim, edge_input_dim, hidden_dim, layers, negative_slope, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def run_inference(model, data_loader, n_steps, greedy, T):
    model.eval()  # Set the model to evaluation mode
    results = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch.to(device)
            actions, tour_logp = model(batch, n_steps, greedy, T)
            print("Actions: ", actions)
            print("Tour Log Probabilities: ", tour_logp)
            
            # Adding the depot {0} at the end of every route
            depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            actions = torch.cat([actions, depot_tensor], dim=1)
            
            # Convert actions tensor to list
            actions_list = actions.cpu().numpy().tolist()
            actions_str = ','.join(map(str, actions_list))
            
            # Get the reward value for the batch
            reward = compute_reward(actions, batch)
            results.append((reward.item(), actions_str))
    return results

def main():
    # Define paths
    model_path = r"model_checkpoints\99\actor.pt"
    # model_path = r"model_checkpoints_2\99\actor.pt"
    # data_path = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_20.CSV'
    data_path = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\TSP_test_20_100.CSV'
    # data_path = r"instances/train/train_10_100.CSV"
    
    
    #Params
    node_input_dim = 1
    edge_input_dim = 1
    hidden_dim = 128
    layers = 3
    negative_slope = 0.2
    dropout = 0.6
    n_steps = 100
    greedy = True
    T = 2.5 # Temperature for softmax based on Kun et al. (2021)
    batch_size = 1

    # Prepare the data
    data_loader = prepare_data(data_path, batch_size)
    
    # Load the pre-trained model
    model = load_model(node_input_dim, edge_input_dim, hidden_dim, layers, negative_slope, dropout, model_path, device)

    # Run inference
    results = run_inference(model, data_loader, n_steps, greedy, T)
    
    # Load the original CSV
    df = pd.read_csv(data_path)
    
    # Assuming there's a one-to-one mapping between the rows in the CSV and the results
    if len(df['InstanceID'].unique()) == len(results):
        result_idx = 0
        for instance_id in df['InstanceID'].unique():
            first_occurrence_idx = df.index[df['InstanceID'] == instance_id].tolist()[0]
            df.at[first_occurrence_idx, 'GAT_Value'] = results[result_idx][0]
            df.at[first_occurrence_idx, 'GAT_Solution'] = results[result_idx][1]
            result_idx += 1
    else:
        print("Warning: The number of results does not match the number of unique instance IDs in the CSV file.")
    
    # Save the updated DataFrame back to CSV
    df.to_csv(data_path, index=False)
    print(f"Results saved to {data_path}")

if __name__ == "__main__":
    main()
