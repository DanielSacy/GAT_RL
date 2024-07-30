import torch
import os
import logging
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# CUDA is used if available, otherwise CPU is used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def prepare_data(data_path, batch_size=1):
    IG = InstanceGenerator()
    data_loader = IG.get_dataloader(data_path, batch_size=batch_size)
    capacity = data_loader.dataset[0].capacity
    return data_loader, capacity

def load_model(model_path, device, node_input_dim=1, edge_input_dim=1, hidden_dim=128, dropout=0.6, layers=2, heads=8, capacity=1000, T=1.0):
    model = Model(node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity, T).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def run_inference(model, data_loader, n_steps, greedy, T):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            actions, tour_logp, depot_visits = model(batch, n_steps, greedy, T)
            print("Actions: ", actions)
            print("Tour Log Probabilities: ", tour_logp)
            print("Depot Visits: ", depot_visits)

def main():
    # Define paths
    model_path = r"model_checkpoints\99\actor.pt"
    data_path = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\test_20nodes_TSP_100.CSV'
    
    
    #Params
    node_input_dim = 1
    edge_input_dim = 1
    hidden_dim = 128
    dropout = 0.6
    layers = 2
    heads = 8
    n_steps = 1
    greedy = True
    T = 2.5 # Temperature for softmax based on Kun et al. (2021)
    batch_size = 10

    # Prepare the data
    data_loader, capacity = prepare_data(data_path, batch_size)
    for batch in data_loader:
        print(batch)
    
    # Load the pre-trained model
    model = load_model(model_path, device, node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity, T)

    # Run inference
    run_inference(model, data_loader, n_steps, greedy, T)

if __name__ == "__main__":
    main()
