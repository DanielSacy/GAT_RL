import logging
import torch
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model
from time import time

def run_model_on_data(filename, batch_size=1):
    """Runs the eGAT encoder on the data from the given CSV file and instance ID.
    Returns the node, edge, and graph embeddings as tensors.
    Later will run the decoder on the graph embedding to get the solution.
    
    IT WILL BE TURNED INTO A MODEL CLASS LATER.
    """
    
    # Configuring the logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    ''' 
    CUDA is used if available, otherwise CPU is used.
    '''
    # print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Running on device: {device}")
    
    '''
    Load the data from the CSV file and convert it to a data object.
    '''
    IG = InstanceGenerator()
    data_loader = IG.get_dataloader(filename, batch_size=batch_size)
    
    '''
    Initialize the Model and run it on the data object.
    '''
    node_input_dim=1 
    edge_input_dim=1
    hidden_dim=16 
    dropout=0.6 
    layers=1 
    heads=1 
    capacity = data_loader.dataset[0].capacity
    greedy = True
    T = 2.0 # Temperature for softmax based on Kun et al. (2021)
    
    model = Model(node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity, T)
    model = model.to(device)
    
    n_steps = 30 # Number of steps for the decoder
    
    all_actions = []
    all_log_p = []

    data_count = 0
    for data in data_loader:
        data_count += 1
        data = data.to(device)
        actions, log_p, depot_visits = model(data, n_steps, greedy=greedy, T=1)
        all_actions.append(actions)
        all_log_p.append(log_p)
    print(f"Data count: {data_count*batch_size}")

    return actions, log_p, depot_visits, all_actions, all_log_p

if __name__ == "__main__":
    filename = 'instance_creator\instance_data.csv'
    # instance_id = '10_1'
    batch_size = 10
    
    start = time()
    actions, log_p, depot_visits, all_actions, all_log_p = run_model_on_data(filename, batch_size=batch_size)

    print(f'Actions: {actions}')
    # print(f'Log probability: {log_p}')
    print(f'Depot visits: {depot_visits}')
    print('All actions:\n' + '\n'.join([f'{action}\n' for action in all_actions]))
    # print(f'All log probability: {all_log_p}')
    
    end = time()
    print(f"Time elapsed: {(end - start):.3} seconds")