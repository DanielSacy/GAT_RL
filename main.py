# Import necessary libraries for data manipulation and machine learning tasks
import pandas as pd  # Library for efficient data analysis and manipulation

# Importing PyTorch library for building neural networks
import torch  # Library for building and training neural networks

# Importing operating system module for interacting with the file system
import os  # Module for working with files, directories, and paths

# Importing logging module for creating logs during execution
import logging  # Module for creating log messages during runtime

# Importing custom modules for instance generation and model definition
from instance_creator.InstanceGenerator import InstanceGenerator  # Custom module for generating instances
from model.Model import Model  # Custom module for defining models

# Importing custom modules for computing rewards and pairwise costs in the RL algorithm
from RL.Compute_Reward import compute_reward  # Custom function for computing rewards
from RL.Pairwise_cost import pairwise_cost  # Custom function for computing pairwise costs

# Configure logging settings to display log messages with specific format
logging.basicConfig(level=logging.DEBUG,  # Set logging level to DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
                    datefmt='%Y-%m-%d %H:%M:%S')  # Date and time format

# Check if CUDA is available on the system; use GPU for computation if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to CUDA or CPU based on availability
logging.info(f"Running on device: {device}")  # Log message indicating the device used for execution

# Define a function to prepare data for training
def prepare_data(data_path, batch_size=1):
    """
    Prepare data loader from CSV file.

    Args:
        data_path (str): Path to the CSV file.
        batch_size (int): Number of samples in each batch. Default is 1.

    Returns:
        torch.utils.data.DataLoader: A data loader object containing the prepared data.
    """
    # Create an instance generator, which will be used to get a data loader from the CSV file
    IG = InstanceGenerator()
    
    # Get a data loader from the CSV file using the instance generator
    data_loader = IG.get_dataloader_CSV(data_path, batch_size=batch_size)
    
    # Return the prepared data loader, which can be used for training or evaluation
    return data_loader

def run_inference(model, data_loader, n_steps, greedy, T):
    """
    Run inference on the model using the provided data loader.

    Args:
        model (Model): The model to be used for inference.
        data_loader (torch.utils.data.DataLoader): A data loader object containing the input data.
        n_steps (int): The number of steps to run the model for.
        greedy (bool): Whether to use a greedy policy or not.
        T (float): The temperature parameter.

    Returns:
        list: A list of tuples containing the reward value and action string for each batch.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize an empty list to store the results
    results = []
    
    # Enable inference mode for the PyTorch session
    with torch.inference_mode():
        # Iterate over each batch in the data loader
        for batch in data_loader:
            # Move the batch to the device (GPU or CPU)
            batch = batch.to(device)
            
            # Run the model on the batch and get the actions and tour log probability
            actions, tour_logp = model(batch, n_steps, greedy, T)
            
            # Print the tour log probabilities for debugging purposes
            print("Tour Log Probabilities: ", tour_logp)
            
            # Get the reward value for the batch using the pairwise cost function
            reward = pairwise_cost(actions, batch)
            
            # Add the depot {0} at the end of every route in the actions tensor
            depot_tensor = torch.zeros(actions.size(0), 1, dtype=torch.long, device=actions.device)
            actions = torch.cat([depot_tensor, actions, depot_tensor], dim=1)
            
            # Print the actions for debugging purposes
            print("Actions: ", actions)
            
            # Convert the actions tensor to a list and join it into a string
            actions_list = actions.cpu().numpy().tolist()
            actions_str = ','.join(map(str, actions_list))
            
            # Append the reward value and action string to the results list
            results.append((reward.item(), actions_str))
    # Return the results list
    return results

def main():
    """
    Main function that loads the model and runs inference on the given data.
    
    Parameters:
    None
    
    Returns:
    None
    """
    
    # Define paths
    model_path = r"actor.pt"
    # model_path = r"KunLei_actor.pt"
    data_path = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\validation\Nodes10_Instances100_EUCLIDEAN.csv"
    
    # Parameters
    node_input_dim = 3
    edge_input_dim = 1
    hidden_dim = 128
    edge_dim = 16
    layers = 4
    negative_slope = 0.2
    dropout = 0.6
    
    # Model parameters
    n_steps = 100  # Number of steps for the model to run
    greedy = True   # Whether to use greedy policy or not
    T = 2.5        # Temperature for softmax based on Kun et al. (2021)
    
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
            
            # Save the results to the DataFrame
            df.at[first_occurrence_idx, 'GAT_Baseline'] = results_baseline[result_idx][0]
            df.at[first_occurrence_idx, 'GAT_BL_Solution'] = results_baseline[result_idx][1]
            df.at[first_occurrence_idx, 'GAT_Trained'] = results_trained[result_idx][0]
            df.at[first_occurrence_idx, 'GAT_Trained_Solution'] = results_trained[result_idx][1]
            
            result_idx += 1
    else:
        print("Warning: The number of results does not match the number of unique instance IDs in the CSV file.")
    
    # Save the updated DataFrame back to CSV
    df.to_csv(data_path, index=False)
    print(f"Results saved to {data_path}")

if __name__ == "__main__":
    main()
