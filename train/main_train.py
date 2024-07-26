import torch
import os

from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model
from src_batch.RL.Rollout_Baseline import RolloutBaseline
from src_batch.train.train_model import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main_train():
    
    # Define the folder and filename for the model checkpoints
    folder = 'model_checkpoints'
    filename = 'actor.pt'

    # Create dataset
    train_dataset = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_100.CSV'
    validation_dataset = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\validation\val_100.CSV'
    
    # Create dataloaders
    IG = InstanceGenerator()
    batch_size = 2
    data_loader = IG.get_dataloader(train_dataset, batch_size=batch_size)
    validation_loader = IG.get_dataloader(validation_dataset, batch_size=batch_size)
    
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Model parameters
    node_input_dim = 1
    edge_input_dim = 1
    hidden_dim = 128
    dropout = 0.6
    layers = 2
    heads = 8
    capacity = data_loader.dataset[0].capacity
    n_steps = 100
    lr = 1e-4
    # greedy = False
    T = 2.5 #1.0
    # Define hyperparameters
    num_epochs = 1
    n_rollouts = 1
    
    # Instantiate the Model and the RolloutBaseline
    model = Model(node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity, T).to(device)
    rol_baseline = RolloutBaseline(model, n_rollouts)
    # rol_baseline = RolloutBaseline(model, data_loader, n_steps, T, epoch=0).to(device)
    
    # Call the train function
    train(model, rol_baseline, data_loader, validation_loader, folder, filename, lr, n_steps, num_epochs, T)

if __name__ == "__main__":
    main_train()


# import torch
# from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
# from src_batch.model.Model import Model
# from src_batch.RL.Rollout_Baseline import RolloutBaseline
# from src_batch.train.train_model import Train

# if __name__ == "__main__":
#     filename = 'instance_creator/instance_data.csv'
#     # filename = 'instance_creator/instance_data_100.csv'
    
#     batch_size = 10
    
#     IG = InstanceGenerator()
#     data_loader = IG.get_dataloader(filename, batch_size=batch_size)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     node_input_dim = 1
#     edge_input_dim = 1
#     hidden_dim = 128
#     dropout = 0.6
#     layers = 2
#     heads = 8
#     capacity = data_loader.dataset[0].capacity
#     n_steps = 10000
#     lr = 1e-4
#     # greedy = False
#     T = 1.0 #2.5
    
#     model = Model(node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity).to(device)
    
#     baseline = RolloutBaseline(model, n_rollouts=30)
    
#     trainer = Train(model, data_loader, device, baseline, n_steps, lr=lr)
#     trainer.train(n_epochs=10)