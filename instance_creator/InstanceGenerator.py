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
    def __init__(self, n_customers=5, n_vehicles=4, max_demand=10, max_distance=20, random_seed=None):
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.max_demand = max_demand
        self.max_distance = max_distance
        
        np.random.seed(random_seed)
    
    @staticmethod  
    def compute_mst_route_value(route, distance):
        total_value = 0
        for i in range(len(route) - 1):
            total_value += distance[route[i], route[i + 1]]
        # Add the return to the depot
        total_value += distance[route[-1], route[0]]
        return total_value

    def instanSacy(self):
        '''
        If I want to go full EUCLIDEAN, I can use the following code:
        
        No = set(np.arange(1, self.n_customers + 1))  # Set of customers
        N = No | {0}  # Customers + depot
        
        # Generate random points for the depot (node 0) and customers
        points = np.random.uniform(0, 1, (len(N), 2))  # Points in [0, 1] x [0, 1]
        # Compute the Euclidean distance matrix for all points
        dist_matrix = DM(points, points)

        # Create the distance dictionary based on the Euclidean distances
        distance_dict = {(i, j): dist_matrix[i][j] for i in range(len(N)) for j in range(len(N))}
        '''
        
        No = set(np.arange(1, self.n_customers + 1))  # Set of customers
        N = No | {0}  # Customers + depot

        Arcs = [(i, j) for i in N for j in N if i != j]  # Set of arcs between the nodes

        # demand = {i: 0 if i not in No else np.random.randint(1, self.max_demand+1) for i in N} # Demand per customer
        demand = {i: 0 if i not in No else np.random.uniform(0.1, self.max_demand/10) for i in N} # Demand per customer

        M = list(np.arange(1, self.n_vehicles + 1))  # Set of vehicles

        load_capacity = 3.0  # Load capacity per vehicle
        # load_capacity = 500  # For TSP simulation

        distance = {(i, j): 0 if i == j else np.random.uniform(0.1, self.max_distance/10) for i in N for j in N}  # Distance between nodes
        # distance =  {(i,j):(np.random.randint(1, self.max_distance+1)) for i,j in Arcs}
        
        # Compute the distance matrix for MST computation and pairwise distance computation
        distance_matrix = np.zeros((len(N), len(N)))
        for (i, j), dist in distance.items():
            distance_matrix[i][j] = dist
            
        # Compute the minimum spanning tree (MST) for baseline route
        mst_bl = mst(csr_matrix(distance_matrix)).toarray().astype(int)
        mst_baseline_route = (depth_first_order(mst_bl, i_start=0, directed=False, return_predecessors=False))
        mst_baseline_value = InstanceGenerator.compute_mst_route_value(mst_baseline_route, distance)
        
        return No, N, M, demand, load_capacity, distance, mst_baseline_value, mst_baseline_route, distance_matrix

    def instance_to_data(self):
        No, N, M, demand, load_capacity, distance, mst_baseline_value, mst_baseline_route, distance_matrix = self.instanSacy()

        node_features = torch.tensor([0] + [demand[i] for i in No], dtype=torch.float).unsqueeze(1)
        edge_index = torch.tensor([[i, j] for i in N for j in N if i != j], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([distance[(i.item(), j.item())] for i, j in edge_index.t()], dtype=torch.float).unsqueeze(1)

        mst_route = torch.tensor(mst_baseline_route, dtype=torch.long)
        mst_value = torch.tensor([mst_baseline_value], dtype=torch.float)
        
        capacity = torch.tensor([load_capacity], dtype=torch.float)
        
        distance_matrix_tensor = torch.tensor(distance_matrix, dtype=torch.float)
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, demand=node_features, capacity=capacity, mst_value=mst_value, mst_route=mst_route, distance_matrix=distance_matrix_tensor)
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
        for dataset in data_loader:
            print(f'dataset.x: {dataset.x}, dataset.edge_index: {dataset.edge_index}, dataset.edge_attr: {dataset.edge_attr}, dataset.demand: {dataset.demand}, dataset.capacity: {dataset.capacity}, dataset.mst_value: {dataset.mst_value}, dataset.mst_route: {dataset.mst_route}, dataset.distance_matrix: {dataset.distance_matrix}')
        return data_loader

    def generate_and_save_instances(self, data_list, filename):
        all_instances = []
        for instance_num, data in enumerate(data_list, start=1):
                node_demands = data.x.numpy().flatten()
                edge_indices = data.edge_index.numpy().T
                edge_distances = data.edge_attr.numpy().flatten()
                capacity = data.capacity.numpy()[0]
                mst_value = data.mst_value.numpy()[0]
                mst_route = data.mst_route.numpy()
                distance_matrix = data.distance_matrix.numpy()


                instance_df = pd.DataFrame({
                    'InstanceID': f'{self.n_customers}_{instance_num}',
                    'FromNode': edge_indices[:, 0],
                    'ToNode': edge_indices[:, 1],
                    'Distance': edge_distances,
                    'Demand': np.repeat(node_demands, len(edge_distances) // len(node_demands)),
                    'Capacity': np.repeat(capacity, len(edge_distances)),
                    'MstValue': np.repeat(mst_value, len(edge_distances)),
                    'MstRoute': np.repeat(mst_route, len(edge_distances) // len(mst_route), axis=0),
                    'DistanceMatrix': distance_matrix.flatten()
                })

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
            
            node_features = demands
            edge_index = torch.tensor(instance_df[['FromNode', 'ToNode']].values.T, dtype=torch.long)
            edge_attr = torch.tensor(instance_df['Distance'].values, dtype=torch.float).unsqueeze(1)
            
            capacity = torch.tensor([instance_df['Capacity'].values[0]], dtype=torch.float)  # Single capacity value for the instance
            
            mst_value = torch.tensor(instance_df['MstValue'].values[0], dtype=torch.float)
            mst_route = instance_df['MstRoute'].values[0]
            
            # distance_matrix = torch.tensor(instance_df['DistanceMatrix'].values.reshape(len(node_features), len(node_features)))
            
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, demand=demands, capacity=capacity, mst_value=mst_value, mst_route=mst_route)

            data_list.append(data)    
        return data_list

    def get_dataloader_CSV(self, filename, batch_size=1):
        data_list = self.csv_to_data_list(filename)
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        return data_loader

