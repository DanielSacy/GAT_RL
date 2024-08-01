import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Tuple


class InstanceGenerator:
    def __init__(self, n_customers=5, n_vehicles=2, max_demand=100, max_distance=200):
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.max_demand = max_demand
        self.max_distance = max_distance

    def instanSacy(self: "InstanceGenerator"):
        """
        Creates an instance of the Vehicle Routing Problem (VRP).

        Args:
            self: The instance generator itself.

        Returns:
            A tuple containing:
                - No: Set of customers.
                - N: Set of nodes (customers + depot).
                - M: List of vehicles.
                - demand: Dictionary mapping node to demand.
                - load_capacity: Load capacity per vehicle.
                - distance: Dictionary mapping arc to distance.
        """
        No = set(np.arange(1, self.n_customers + 1))  # Set of customers
        N = No | {0}  # Customers + depot

        Arcs = [(i, j) for i in N for j in N if i != j]  # Set of arcs between the nodes

        demand = {
            i: 0 if i not in No else int(np.random.randint(50, self.max_demand, 1)[0])
            for i in N
        }  # Demand per customer

        M = list(np.arange(1, self.n_vehicles + 1))  # Set of vehicles

        load_capacity = 1000  # Load capacity per vehicle
        # load_capacity = 10000  # For TSP simulation

        distance = {
            (i, j): int(np.random.randint(50, self.max_distance, 25)[0])
            for i, j in Arcs
        }
        # distance = {(i, j): 100 for i in N for j in N if i != j}  # Constant distance for simplicity

        return No, N, M, demand, load_capacity, distance

    def instance_to_data(self: "InstanceGenerator") -> Data:
        """
        Converts an instance to a PyTorch dataset.

        Args:
            self: The instance generator itself.

        Returns:
            A PyTorch dataset representing the instance.
        """
        No, N, M, demand, load_capacity, distance = self.instanSacy()

        node_features = torch.tensor(
            [0] + [demand[i] for i in No], dtype=torch.float
        ).unsqueeze(1)
        edge_index = (
            torch.tensor([[i, j] for i in N for j in N if i != j], dtype=torch.long)
            .t()
            .contiguous()
        )
        edge_attr = torch.tensor(
            [distance[(i.item(), j.item())] for i, j in edge_index.t()],
            dtype=torch.float,
        ).unsqueeze(1)

        capacity = torch.tensor([load_capacity], dtype=torch.float)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            demand=node_features,
            capacity=capacity,
        )

        return data

    def generate_and_save_instances(
        self: "InstanceGenerator",
        instances_config: List[dict],
        filename: str = "instance_data.csv",
    ):
        """
        Generates and saves multiple instances of the Vehicle Routing Problem (VRP).

        Args:
            self: The instance generator itself.
            instances_config: List of configuration dictionaries, each containing:
                - n_customers: Number of customers.
                - max_demand: Maximum demand per customer.
                - max_distance: Maximum distance between nodes.
                - num_instances: Number of instances to generate for this configuration.
            filename: Name of the output CSV file.
        """
        all_instances = []
        capacities = []
        for config in instances_config:
            self.n_customers, self.max_demand, self.max_distance = (
                config["n_customers"],
                config["max_demand"],
                config["max_distance"],
            )
            for instance_num in range(1, config["num_instances"] + 1):
                data = self.instance_to_data()
                node_demands = data.x.numpy().flatten()
                edge_indices = data.edge_index.numpy().T
                edge_distances = data.edge_attr.numpy().flatten()
                capacity = data.capacity.numpy()[0]

                instance_df = pd.DataFrame(
                    {
                        "InstanceID": f"{self.n_customers}_{instance_num}",
                        "FromNode": edge_indices[:, 0],
                        "ToNode": edge_indices[:, 1],
                        "Distance": edge_distances,
                        "Demand": np.repeat(
                            node_demands, len(edge_distances) // len(node_demands)
                        ),
                    }
                )
                capacities.append(capacity)

                all_instances.append(instance_df)

        full_df = pd.concat(all_instances, ignore_index=True)
        full_df["Capacity"] = (
            pd.Series(capacities).repeat(len(full_df) // len(capacities)).values
        )
        full_df.to_csv(filename, index=False)
        print(f"All instances saved to {filename}")

    def csv_to_data_list(
        self: "InstanceGenerator", filename="instance_data.csv"
    ) -> List[Data]:
        """
        Converts a CSV file into a list of PyTorch datasets.

        Args:
            self: The instance generator itself.
            filename: Name of the input CSV file.

        Returns:
            A list of PyTorch datasets, each representing an instance of the Vehicle Routing Problem (VRP).
        """
        df = pd.read_csv(filename)
        data_list = []
        for instance_id in df["InstanceID"].unique():
            instance_df = df[df["InstanceID"] == instance_id]
            if instance_df.empty:
                continue

            demands = instance_df.groupby("FromNode")["Demand"].first().to_dict()
            demands = torch.tensor(
                [demands[node] for node in sorted(demands.keys())], dtype=torch.float
            ).unsqueeze(1)

            node_features = demands
            edge_index = torch.tensor(
                instance_df[["FromNode", "ToNode"]].values.T, dtype=torch.long
            )
            edge_attr = torch.tensor(
                instance_df["Distance"].values, dtype=torch.float
            ).unsqueeze(1)

            capacity = torch.tensor(
                [instance_df["Capacity"].values[0]], dtype=torch.float
            )  # Single capacity value for the instance

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                demand=demands,
                capacity=capacity,
            )
            data_list.append(data)
        return data_list

    def get_dataloader(self: "InstanceGenerator", filename: str, batch_size: int = 1):
        """
        Creates a PyTorch DataLoader from the given list of datasets.

        Args:
            self: The instance generator itself.
            filename: Name of the csv file to be loaded
            batch_size: Number of instances to include in each batch. Defaults to 1.

        Returns:
            A PyTorch DataLoader that can be used for training or testing.
        """
        data_list = self.csv_to_data_list(filename)
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        return data_loader
