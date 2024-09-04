import json
from math import ceil
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = [data.to('cuda') for data in data_list]  # Move data to GPU

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
class InstanceGenerator:
    def __init__(self, n_customers=5, n_vehicles=4, max_demand=10, max_distance=20, random_seed=42):
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.max_demand = max_demand
        self.max_distance = max_distance
        
        np.random.seed(random_seed)
    
    def euclidean_distance(p1, p2):
        return (np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def instanSacy(self):
        No = set(np.arange(1, self.n_customers + 1))  # Set of customers
        N = No | {0}  # Customers + depot
        
        ''''''
        # GOING EUCLIDEAN
        # coordinates = {i: np.random.randint(0, self.max_distance+1, size=2) for i in N}
        coordinates = {i: np.random.randint(0, self.max_distance+1, size=2)/100 for i in N}
        
        # Create the distance matrix using Euclidean distance
        distance = {(i, j): 0 if i == j else InstanceGenerator.euclidean_distance(coordinates[i], coordinates[j]) for i in N for j in N}
        ''''''

        demand = {i: 0 if i not in No else np.random.randint(1, self.max_demand+1)/10 for i in N} # Demand per customer

        M = list(np.arange(1, self.n_vehicles + 1))  # Set of vehicles

        load_capacity = 3  # Load capacity per vehicle
        
        return N, demand, load_capacity, distance, coordinates

    def instance_to_data(self):
        N, demand, load_capacity, distance, coordinates = self.instanSacy()

        # Node features only coordinates
        node_features = torch.tensor([coordinates[i].tolist() for i in N], dtype=torch.float)
        edge_index = torch.tensor([[i, j] for i in N for j in N ], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([distance[(i.item(), j.item())] for i, j in edge_index.t()], dtype=torch.float).unsqueeze(1)
        
        demand = torch.tensor([demand[i] for i in N], dtype=torch.float).unsqueeze(1)
        capacity = torch.tensor([load_capacity], dtype=torch.float)
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, demand=demand, capacity=capacity)
        return data
    
    def get_dataloader_memory(self, instances_config, batch_size, save_to_csv=False, filename=None):
        data_list = []
        for config in instances_config:
            self.n_customers, self.max_demand, self.max_distance = config['n_customers'], config['max_demand'], config['max_distance']
            for _ in range(1, config['num_instances'] + 1):
                data = self.instance_to_data()
                data_list.append(data)
            
            if save_to_csv and filename:
                self.generate_and_save_instances(data_list, filename)
        
        
        
        # in_memory_dataset = InMemoryDataset(data_list)
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        # for dataset in data_loader:
        #     print(f'dataset.x: {dataset.x}, dataset.edge_index: {dataset.edge_index}, dataset.edge_attr: {dataset.edge_attr}, dataset.demand: {dataset.demand}, dataset.capacity: {dataset.capacity}, dataset.mst_value: {dataset.mst_value}, dataset.mst_route: {dataset.mst_route}, dataset.distance_matrix: {dataset.distance_matrix}')
        return data_loader

    def generate_and_save_instances(self, data_list, filename):
        all_instances = []
        for instance_num, data in enumerate(data_list, start=1):
            node_features = np.round(data.x.numpy().tolist(), 2)
            node_demands = data.demand.numpy().flatten()
            edge_indices = data.edge_index.numpy().T
            edge_distances = np.round(data.edge_attr.numpy().flatten(), 2)
            capacity = data.capacity.numpy()[0]
            # distance_matrix = data.distance_matrix.numpy().flatten()
            # distance_matrix = data.distance_matrix.numpy()
            
            # Serialize all data using json.dumps
            serialized_capacity = json.dumps(capacity.tolist())
            serialized_node_features = json.dumps(node_features.tolist())


            instance_df = pd.DataFrame({
                'InstanceID': f'{self.n_customers}_{instance_num}',
                'FromNode': edge_indices[:, 0],
                'ToNode': edge_indices[:, 1],
                'Distance': edge_distances,
                'Distance_SCIP': edge_distances * 100,
                'NodeFeatures': [serialized_node_features] * len(edge_distances),  
                'Demand': np.repeat(node_demands, len(edge_distances) // len(node_demands)),
                'Capacity': [''] * len(edge_distances),  # Empty for all rows except the first
            })

            instance_df.at[0, 'Capacity'] = serialized_capacity

            all_instances.append(instance_df)

        full_df = pd.concat(all_instances, ignore_index=True)
        full_df.to_csv(filename, index=False)
        print(f"All instances saved to {filename}")
    
    def csv_to_data_list(self, filename):
        df = pd.read_csv(filename)
        
        data_list = []
        for instance_id in df['InstanceID'].unique():
            instance_df = df[df['InstanceID'] == instance_id]
            if instance_df.empty:
                break
            
            demands = instance_df.groupby('FromNode')['Demand'].first().to_dict()
            demands = torch.tensor([demands[node] for node in sorted(demands.keys())], dtype=torch.float).unsqueeze(1)
            
            # Deserialize node features from the first row of the NodeFeatures column
            node_features = json.loads(instance_df['NodeFeatures'].iloc[0])
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(instance_df[['FromNode', 'ToNode']].values.T, dtype=torch.long)
            edge_attr = torch.tensor(instance_df['Distance'].values, dtype=torch.float).unsqueeze(1)
            
            capacity = instance_df['Capacity'].iloc[0]
            capacity = torch.tensor([capacity], dtype=torch.float)
            
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, demand=demands, capacity=capacity)

            data_list.append(data)    
        return data_list

    def get_dataloader_CSV(self, filename, batch_size=1):
        data_list = self.csv_to_data_list(filename)
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        return data_loader