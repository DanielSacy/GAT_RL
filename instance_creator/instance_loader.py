# Import the necessary libraries
import logging
from instance_creator.InstanceGenerator import InstanceGenerator

# Configure logging to display debug messages and format them with timestamp, name, level, and message
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def instance_loader(config, batch_size, save_to_csv):
    """
    Loads instances based on the provided configuration.
    
    Args:
        config (list): A list of dictionaries containing instance generation parameters.
        batch_size (int): The number of instances to generate in each batch.
        save_to_csv (bool): Whether to save the generated instances to a CSV file.
        
    Returns:
        data_loader: An object that provides access to the generated instances.
    """
    
    # Extract values for x and y from the first configuration in the list
    x = config[0]['n_customers']  # Number of customers
    
    # Extract the number of instances from the first configuration
    y = config[0]['num_instances']
    
    # Create the filename based on the extracted parameters
    filename = f'instances\\Nodes{x}_Instances{y}_EUCLIDEAN.csv'
    
    # Instantiate the InstanceGenerator class with a fixed random seed for reproducibility
    generator = InstanceGenerator(random_seed=42)   
    
    # Generate the instances based on the configuration and store them in a data loader object
    data_loader = generator.get_dataloader_memory(config, batch_size, save_to_csv, filename=filename)

    # Log messages if saving to CSV or creating the data loader
    if save_to_csv:
        logging.info(f"Saving instances to {filename}")      
    logging.info(f"Data loader created with {len(data_loader)* batch_size} instances")
        
    return data_loader

# Check if this script is being run directly (i.e., not imported as a module)
if __name__ == '__main__':
    # Define the configuration for instance generation
    config = [
        # {'n_customers': 3, 'max_demand': 30, 'max_distance': 40, 'num_instances': 2}
        {'n_customers': 5, 'max_demand': 10, 'max_distance': 40, 'num_instances': 1}
        # Add more configurations as needed
    ]
    
    # Set the batch size and CSV saving parameters
    batch_size = 1
    save_to_csv = False
    
    # Load instances using the provided configuration and parameters
    instance_loader(config, batch_size, save_to_csv)