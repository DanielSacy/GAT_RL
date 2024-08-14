import time
import torch
import os
import logging

from src_batch.model.Model import Model
from src_batch.train.train_model import train
from src_batch.instance_creator.instance_loader import instance_loader



logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def main_train():
    logging.info("Starting training pipeline")
    # Define the folder and filename for the model checkpoints
    # folder = 'model_checkpoints_2' # Best model so far
    folder = 'model_checkpoints'
    filename = 'actor.pt'

    # Create dataset
    '''DEBUG'''
    # train_dataset = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\debug_1_2instances.CSV"
    # train_dataset = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\debug_4_2instances.CSV"
    # train_dataset = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\debug_10.CSV"
    '''TRAIN'''
    train_dataset = r"D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_20_50000.CSV"
    validation_dataset = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\validation\val_10_100.CSV'
    
    # Define the configurations for the instances
    config = [
    {'n_customers': 20, 'max_demand': 10, 'max_distance': 20, 'num_instances': 24000}
    # Add more configurations as needed
    ]
    
    logging.info("Creating dataloaders")
    # Create dataloaders
    batch_size = 16
    save_to_csv = False
    data_loader = instance_loader(config, batch_size, save_to_csv)   
    
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Model parameters
    node_input_dim = 1
    edge_input_dim = 1
    hidden_dim = 64
    layers = 4
    negative_slope = 0.2
    dropout = 0.6
    n_steps = 100
    lr = 1e-4
    # greedy = False
    T = 2.5 #1.0

    num_epochs = 1
    
    logging.info("Instantiating the model")
    # Instantiate the Model and the RolloutBaseline
    model = Model(node_input_dim, edge_input_dim, hidden_dim, layers, negative_slope, dropout).to(device)
    
    logging.info("Calling the train function")
    # Call the train function
    train(model, data_loader, folder, filename, lr, n_steps, num_epochs, T)

if __name__ == "__main__":
    pipeline_start = time.time()
    main_train()
    pipeline_end = time.time()
    logging.info(f"Pipeline execution time: {pipeline_end - pipeline_start} seconds")