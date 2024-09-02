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
    """
    A custom dataset class for storing data in memory, primarily for use with PyTorch Geometric.
    
    This class is designed to hold a list of pre-computed graphs (Data objects), which are stored on the GPU for efficient computation.
    
    Attributes:
        data_list (list): A list of Data objects representing the graphs to be loaded from memory.
    """
    def __init__(self, data_list):
        # Initialize the dataset by moving each graph in the data list to the GPU
        self.data_list = [data.to('cuda') for data in data_list]  # Move data to GPU

    def __len__(self):
        # Return the total number of graphs in the dataset (i.e., the length of the data list)
        return len(self.data_list)

    def __getitem__(self, idx):
        # Return the graph at the specified index in the data list
        return self.data_list[idx]
    
class InstanceGenerator:
    """
    A class for generating instances of a vehicle routing problem.
    
    Attributes:
        n_customers (int): The number of customers to include in each instance.
        n_vehicles (int): The number of vehicles to use in each instance.
        max_demand (int): The maximum demand value for any customer.
        max_distance (int): The maximum distance between two points.
        random_seed (int): The seed for the random number generator used to generate instances.
    """
    
    def __init__(self, n_customers=5, n_vehicles=4, max_demand=10, max_distance=20, random_seed=42):
        # Set the instance parameters
        self.n_customers = n_customers  # Number of customers in each instance
        self.n_vehicles = n_vehicles  # Number of vehicles in each instance
        self.max_demand = max_demand  # Maximum demand value for any customer
        self.max_distance = max_distance  # Maximum distance between two points
        
        # Set the seed for the random number generator used to generate instances
        np.random.seed(random_seed)
    
    @staticmethod  
    def compute_mst_route_value(route, distance):
        """
        Compute the total value of a route by summing up the distances between consecutive points.
        
        Parameters:
            route (list): The list of point indices in the order they are visited.
            distance (array): The 2D array containing the distances between each pair of points.
            
        Returns:
            float: The total value of the route.
        """
        # Initialize the total value to zero
        total_value = 0
        
        # Iterate over each point in the route, excluding the last one (which returns to the depot)
        for i in range(len(route) - 1):
            # Add the distance between the current and next points to the total value
            total_value += distance[route[i], route[i + 1]]
        
        # Add the distance from the last point back to the depot (the first point)
        total_value += distance[route[-1], route[0]]
        
        # Append the first point to the end of the route, so that it returns to the depot
        route = np.append(route, route[0])
        
        return total_value, route
    
    
    def euclidean_distance(p1, p2):
        """
        Compute the Euclidean distance between two points in a 2D space.
        
        Parameters:
            p1 (tuple): The coordinates of the first point.
            p2 (tuple): The coordinates of the second point.
            
        Returns:
            float: The Euclidean distance between the two points.
        """
        # Calculate the difference in x and y coordinates
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        
        # Use the Pythagorean theorem to compute the Euclidean distance
        return (dx ** 2 + dy ** 2) ** 0.5

    def instanSacy(self):
        No = set(np.arange(1, self.n_customers + 1))  # Set of customers
        N = No | {0}  # Customers + depot
        
        ''''''
        # GOING EUCLIDEAN
        # coordinates = {i: np.random.randint(0, self.max_distance+1, size=2) for i in N}
        coordinates = {i: np.random.randint(0, self.max_distance+1, size=2)/100 for i in N}
        
        # Create the distance matrix using Euclidean distance
        distance = {(i, j): 0 if i == j else InstanceGenerator.euclidean_distance(coordinates[i], coordinates[j]) for i in N for j in N}

        # Convert to a distance matrix format
        distance_matrix = np.zeros((len(N), len(N)))
        for (i, j), dist in distance.items():
            distance_matrix[i][j] = dist
        ''''''
        # Arcs = [(i, j) for i in N for j in N if i != j]  # Set of arcs between the nodes

        demand = {i: 0 if i not in No else np.random.randint(1, self.max_demand+1)/10 for i in N} # Demand per customer

        M = list(np.arange(1, self.n_vehicles + 1))  # Set of vehicles

        load_capacity = 3  # Load capacity per vehicle
        # load_capacity = 500  # For TSP simulation
        
        # IF USING MST BASELINE
        # Compute the minimum spanning tree (MST) for baseline route
        # mst_bl = mst(csr_matrix(distance_matrix)).toarray().astype(int)
        # mst_baseline_route = (depth_first_order(mst_bl, i_start=0, directed=False, return_predecessors=False))
        # mst_baseline_value, mst_baseline_route = InstanceGenerator.compute_mst_route_value(mst_baseline_route, distance)
        
        return N, demand, load_capacity, distance, distance_matrix, coordinates
        # return No, N, M, demand, load_capacity, distance, mst_baseline_value, mst_baseline_route, distance_matrix, coordinates

    def instance_to_data(self):
        """
        Convert an instance to a PyTorch Geometric Data object.
        
        Returns:
            data: A PyTorch Geometric Data object containing the features and edge attributes of the instance.
        """
        N, demand, load_capacity, distance, distance_matrix, coordinates = self.instanSacy()

        # Create node features
        node_features = torch.tensor([coordinates[i].tolist() + [demand[i]] for i in N], dtype=torch.float)

        # Create edge indices and attributes
        edge_index = torch.tensor([[i, j] for i in N for j in N ], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([distance[(i.item(), j.item())] for i, j in edge_index.t()], dtype=torch.float).unsqueeze(1)

        # Create additional attributes
        demand_tensor = torch.tensor([demand[i] for i in N], dtype=torch.float).unsqueeze(1)
        capacity_tensor = torch.tensor([load_capacity], dtype=torch.float)
        distance_matrix_tensor = torch.tensor(distance_matrix, dtype=torch.float)
        
        # Create the PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            demand=demand_tensor,
            capacity=capacity_tensor,
            distance_matrix=distance_matrix_tensor
        )

        return data
    
    def get_dataloader_memory(self, instances_config, batch_size, save_to_csv=False, filename=None):
        """
        Generate a dataset and return a dataloader for in-memory processing.
        
        Args:
            instances_config (list): List of dictionaries containing instance configuration parameters.
            batch_size (int): Number of instances to process per batch.
            save_to_csv (bool): If True, save the generated data to a CSV file.
            filename (str): Name of the CSV file to save to.
        
        Returns:
            dataloader: PyTorch DataLoader object for in-memory processing.
        """
        
        # Initialize an empty list to store the generated instances
        data_list = []
        
        # Iterate over each instance configuration and generate instances
        for config in instances_config:
            self.n_customers, self.max_demand, self.max_distance = config['n_customers'], config['max_demand'], config['max_distance']
            for _ in range(1, config['num_instances'] + 1):
                data = self.instance_to_data()
                data_list.append(data)
        
        # If save_to_csv is True, generate and save the instances to a CSV file
        if save_to_csv and filename:
            self.generate_and_save_instances(data_list, filename)
        
        # Create an InMemoryDataset object from the generated instances
        # in_memory_dataset = InMemoryDataset(data_list) #TODO: check why this was required
        
        # Create a DataLoader object for in-memory processing
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        return data_loader

    def generate_and_save_instances(self, data_list, filename):
        # Serialize all instances to a Pandas DataFrame
        all_instances = []
        for instance_num, data in enumerate(data_list, start=1):
            node_features = data.x.numpy().tolist()
            node_demands = data.demand.numpy().flatten()
            edge_indices = data.edge_index.numpy().T
            edge_distances = data.edge_attr.numpy().flatten()
            capacity = data.capacity.numpy()[0]
            # mst_value = data.mst_value.numpy()[0]
            # mst_route = data.mst_route.numpy()
            
            instance_df = pd.DataFrame({
                'InstanceID': f'{self.n_customers}_{instance_num}',
                'FromNode': edge_indices[:, 0],
                'ToNode': edge_indices[:, 1],
                'Distance': edge_distances,
                'NodeFeatures': [node_features] * len(edge_distances),  # Empty for all rows except the first
                'Demand': np.repeat(node_demands, len(edge_distances) // len(node_demands)),
                'Capacity': [''] * len(edge_distances),  # Empty for all rows except the first
                'MstValue': [''] * len(edge_distances),  # Empty for all rows except the first
                'MstRoute': [''] * len(edge_distances),  # Empty for all rows except the first
            })

            # Insert the serialized values in the first row
            instance_df.at[0, 'DistanceMatrix'] = data.distance_matrix.numpy().tolist()
            instance_df.at[0, 'Capacity'] = capacity.tolist()
            all_instances.append(instance_df)

        full_df = pd.concat(all_instances, ignore_index=True)
        full_df.to_csv(filename, index=False)
        print(f"All instances saved to {filename}")
    
    def csv_to_data_list(self, filename):
        """
        Converts a CSV file into a list of Data objects.
        
        Args:
            filename (str): The path to the CSV file.
        
        Returns:
            data_list (list): A list of Data objects.
        """

        df = pd.read_csv(filename)
        
        data_list = []
        for instance_id in df['InstanceID'].unique():
            # Get the rows that correspond to this instance
            instance_df = df[df['InstanceID'] == instance_id]
            
            if instance_df.empty:
                # If there are no rows, break out of the loop
                break
            
            demands = instance_df.groupby('FromNode')['Demand'].first().to_dict()
            demands = torch.tensor([demands[node] for node in sorted(demands.keys())], dtype=torch.float).unsqueeze(1)
            
            # Deserialize node features from the first row of the NodeFeatures column
            node_features = json.loads(instance_df['NodeFeatures'].iloc[0])
            node_features = torch.tensor(node_features, dtype=torch.float)
            
            edge_index = torch.tensor(instance_df[['FromNode', 'ToNode']].values.T, dtype=torch.long)
            edge_attr = torch.tensor(instance_df['Distance'].values, dtype=torch.float).unsqueeze(1)
            
            # Deserialize the first row values for MST route, distance matrix, capacity, and MST value
            distance_matrix = json.loads(instance_df['DistanceMatrix'].iloc[0])
            capacity = instance_df['Capacity'].iloc[0]
            # mst_value = instance_df['MstValue'].iloc[0]

            # Convert distance_matrix, mst_route, capacity, and mst_value back into tensors
            distance_matrix = torch.tensor(distance_matrix, dtype=torch.float)
            capacity = torch.tensor([capacity], dtype=torch.float)
            
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, demand=demands, capacity=capacity, distance_matrix=distance_matrix)
            data_list.append(data)    
        return data_list

    # Method to create a DataLoader from a CSV file
    def get_dataloader_CSV(self, filename, batch_size=1):
        """
        Loads data from a CSV file and creates a DataLoader.

        Args:
            filename (str): Path to the CSV file.
            batch_size (int): Number of instances to process at once. Defaults to 1.

        Returns:
            DataLoader: A DataLoader containing the loaded data.
        """
        
        # Load data from the CSV file into a list of Data objects
        data_list = self.csv_to_data_list(filename)
        
        # Create a DataLoader with the specified batch size
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        
        return data_loader


    '''
    If I want to go full EUCLIDEAN, I can use the following code:
    
    No = set(np.arange(1, self.n_customers + 1))  # Set of customers
    N = No | {0}  # Customers + depot
    
    # Generate random points for the depot (node 0) and customers
    points = np.random.uniform(0, 1, (len(N), 2))  # Points in [0, 1] x [0, 1]
    # Compute the Euclidean distance matrix for all points
    dist_matrix = DM(points, points)

    # demand = {i: 0 if i not in No else np.random.uniform(0.1, self.max_demand/10) for i in N} # Demand per customer
    # distance = {(i, j): 0 if i == j else np.random.uniform(0.1, self.max_distance/10) for i in N for j in N}  # Distance between nodes
    
    # Create the distance dictionary based on the Euclidean distances
    distance_dict = {(i, j): dist_matrix[i][j] for i in range(len(N)) for j in range(len(N))}
    '''