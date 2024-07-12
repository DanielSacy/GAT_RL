import torch
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator
from src_batch.model.Model import Model
from src_batch.RL.Rollout_Baseline import RolloutBaseline
from src_batch.train.train_model import Train

if __name__ == "__main__":
    # filename = 'instance_creator/instance_data.csv'
    filename = 'instance_creator/instance_data_100.csv'
    
    batch_size = 25
    
    IG = InstanceGenerator()
    data_loader = IG.get_dataloader(filename, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    node_input_dim = 1
    edge_input_dim = 1
    hidden_dim = 128
    dropout = 0.6
    layers = 2
    heads = 8
    capacity = data_loader.dataset[0].capacity
    n_steps = 9
    lr = 0.001
    # greedy = False
    T = 1.0 #2.5
    
    model = Model(node_input_dim, edge_input_dim, hidden_dim, dropout, layers, heads, capacity).to(device)
    
    baseline = RolloutBaseline(model, n_rollouts=30)
    
    trainer = Train(model, data_loader, device, baseline, n_steps=9, lr=lr)
    trainer.train(n_epochs=10)