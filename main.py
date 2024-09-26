import pandas as pd
import torch
import os
import logging
import time
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model
from src_batch.RL.euclidean_cost_eval import euclidean_cost_eval

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# CUDA is used if available, otherwise CPU is used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def prepare_data(data_path, batch_size=1):
    IG = InstanceGenerator()
    data_loader = IG.get_dataloader_CSV(data_path, batch_size=batch_size)
    return data_loader

def run_inference(model, data_loader, n_steps, greedy, T):
    model.eval()  # Set the model to evaluation mode
    results = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch.to(device)
            
            start_time = time.time()
            actions, tour_logp = model(batch, n_steps, greedy, T)
            print("Tour Log Probabilities: ", tour_logp)
            # actions = torch.tensor(actions[0], dtype=torch.long, device=device)
            
            # Get the reward value for the batch
            reward = euclidean_cost_eval(batch.x, actions, batch)
            runtime = round(time.time() - start_time, 4)
            print("Time taken: ", runtime)
            
            # Adding the depot {0} at the end of every route
            depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            actions = torch.cat([depot_tensor, actions, depot_tensor], dim=1)
            print("Actions: ", actions)
            # Convert actions tensor to list
            actions_list = actions.cpu().numpy().tolist()
            actions_str = ','.join(map(str, actions_list))
            
            results.append((reward.item(), actions_str, runtime))
    return results

def main():
    # Define paths
    model_name = 'DiCE'
    # model_path = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\Success\for_dissertation\Greedy_Rollout_Memo_Logging\Greedy_Rollout_Memo_Logging_best_model.pt"
    model_path = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\Success\for_dissertation\DiCE_noSample\DiCE_noSample_best_model.pt"
    # data_path = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\Nodes3_Instances2.csv"
    data_path = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\Dissertation_Nodes20_Instances100.csv"
    
    #Params
    node_input_dim = 3
    edge_input_dim = 1
    hidden_dim = 128
    edge_dim = 16
    layers = 4
    negative_slope = 0.2
    dropout = 0.6
    n_steps = 100
    greedy = True
    T = 2.5 # Temperature for softmax based on Kun et al. (2021)
    batch_size = 1

    # Instantiate the model
    model = Model(node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout).to(device)

    # Prepare the data
    data_loader = prepare_data(data_path, batch_size)
    
    # Run inference for untained model
    results_baseline = run_inference(model, data_loader, n_steps, greedy, T)
    
    # Run inference for trained model
    model.load_state_dict(torch.load(model_path, device))
    results_trained = run_inference(model, data_loader, n_steps, greedy, T)
    
    # Load the original CSV
    df = pd.read_csv(data_path)
    
    # Assuming there's a one-to-one mapping between the rows in the CSV and the results
    if len(df['InstanceID'].unique()) == len(results_baseline):
        result_idx = 0
        for instance_id in df['InstanceID'].unique():
            first_occurrence_idx = df.index[df['InstanceID'] == instance_id].tolist()[0]
            # second_occurrence_idx = df.index[df['InstanceID'] == instance_id].tolist()[1]
            # third_occurrence_idx = df.index[df['InstanceID'] == instance_id].tolist()[2]
            
            df.at[first_occurrence_idx, 'GAT_Baseline'] = results_baseline[result_idx][0]
            df.at[first_occurrence_idx, 'GAT_BL_Solution'] = results_baseline[result_idx][1]
            df.at[first_occurrence_idx, 'GAT_BL_Runtime'] = results_baseline[result_idx][2]
            df.at[first_occurrence_idx, f'{model_name}'] = results_trained[result_idx][0]
            df.at[first_occurrence_idx, f'{model_name}_Solution'] = results_trained[result_idx][1]
            df.at[first_occurrence_idx, f'{model_name}_Runtime'] = results_trained[result_idx][2]
            # df.at[second_occurrence_idx, 'GAT_Baseline'] = "GAT_Trained"
            # df.at[second_occurrence_idx, 'GAT_BL_Solution'] = "GAT_T_Solution"
            # df.at[third_occurrence_idx, 'GAT_Baseline'] = results_trained[result_idx][0]
            # df.at[third_occurrence_idx, 'GAT_BL_Solution'] = results_trained[result_idx][1]
            result_idx += 1
    else:
        print("Warning: The number of results does not match the number of unique instance IDs in the CSV file.")
    
    # Save the updated DataFrame back to CSV
    df.to_csv(data_path, index=False)
    print(f"Results saved to {data_path}")

if __name__ == "__main__":
    main()
