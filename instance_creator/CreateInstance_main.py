# Import the InstanceGenerator class from the InstanceGenerator module.
from InstanceGenerator import InstanceGenerator

def main(config, filename):
    """
    Main function that generates instances based on a configuration and saves them to a CSV file.

    Args:
        config (list): A list of dictionaries containing instance generation configurations.
        filename (str): The path to the CSV file where instances will be saved.
    """
    
    # Create an instance generator object to generate and save instances.
    generator = InstanceGenerator()
    
    # Generate and save instances based on the provided configuration.
    generator.generate_and_save_instances(config, filename)

    print(f"Instances generated based on configuration and saved to respective CSV files.")

# Define a list of configurations for generating instances.
instances_config = [
    {
        'n_customers': 20, 
        'max_demand': 10, 
        'max_distance': 20, 
        'num_instances': 50000
    }
    # Add more configurations as needed.
]

# Set the path to the CSV file where instances will be saved for debugging purposes.
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\debug_1_2instances.CSV'

# Set the path to the CSV file where instances will be saved for training purposes.
filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_20_50000.CSV'

# Set the path to the CSV file where instances will be saved for validation purposes. (Currently not used)
# filename = 

# Set the path to the CSV file where instances will be saved for testing purposes.
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\TSP_test_20_100.CSV'

if __name__ == "__main__":
    # Call the main function with the provided configuration and CSV file path.
    main(instances_config, filename)
