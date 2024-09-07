import time
import datetime
import torch
import os
import logging
from torch.profiler import profile, record_function, ProfilerActivity

from src_batch.model.Model import Model
from src_batch.train.train_model import train
from src_batch.instance_creator.instance_loader import instance_loader

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def main_train():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H")
    logging.info(f"Starting training pipeline: {now}h")
    # Define the folder and filename for the model checkpoints
    # folder = 'model_checkpoints__'
    folder = f'checkpoints\\model_checkpoints_{now}h'
    filename = 'actor.pt'

    # Create dataset
    '''TRAIN'''
    # train_dataset = 
    
    # Define the configurations for the instances
    config = [
    {'n_customers': 20, 'max_demand': 10, 'max_distance': 100, 'num_instances': 768000}
    # {'n_customers': 2, 'max_demand': 10, 'max_distance': 100, 'num_instances': 2}
    # Add more configurations as needed
    ]
    valid_config = [
    # {'n_customers': 2, 'max_demand': 10, 'max_distance': 100, 'num_instances': 1}
     {'n_customers': 20, 'max_demand': 10, 'max_distance': 100, 'num_instances': 10000}
    # Add more configurations as needed
    ]
    # Create dataloaders
    # Sending the data to the device when generating the data
    start_to_load = time.time()
    logging.info("Creating dataloaders")
    batch_size = 512
    save_to_csv = False
    data_loader = instance_loader(config, batch_size, save_to_csv)
    valid_batch_size = 512
    valid_loader = instance_loader(valid_config, valid_batch_size, save_to_csv) 
    end_of_load = time.time()
    logging.info(f"Data loaded in {end_of_load - start_to_load} seconds")
    
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Model parameters
    node_input_dim = 3
    edge_input_dim = 1
    hidden_dim = 128
    edge_dim = 16
    layers = 4
    negative_slope = 0.2
    dropout = 0.6
    n_steps = 100
    lr = 1e-4
    # greedy = False
    T = 2.5 #1.0

    num_epochs = 100
    
    logging.info("Instantiating the model")
    # Instantiate the Model and the RolloutBaseline
    model = Model(node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout)
    
    logging.info("Calling the train function")
    # Call the train function
    train(model, data_loader, valid_loader, folder, filename, lr, n_steps, num_epochs, T)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         train(model, data_loader, folder, filename, lr, n_steps, num_epochs, T)
        
    # logging.info(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    logging.info("Training pipeline finished")

if __name__ == "__main__":
    pipeline_start = time.time()
    main_train()
    pipeline_end = time.time()
    logging.info(f"Pipeline execution time: {pipeline_end - pipeline_start} seconds")