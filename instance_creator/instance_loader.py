import logging
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def instance_loader(config, batch_size, save_to_csv):
    
    
    # Extract values for x and y from the first configuration in the list
    x = config[0]['n_customers']
    y = config[0]['num_instances']

    # Create the filename
    filename = f'instances\\Nodes{x}_Instances{y}.csv'
    
    # Instantiate the InstanceGenerator class
    generator = InstanceGenerator(42)
    
    # Generate the instances based on the configuration
    data_loader = generator.get_dataloader_memory(config, batch_size, save_to_csv, filename=filename)

    if save_to_csv:
        logging.info(f"Saving instances to {filename}")      
    logging.info(f"Data loader created with {len(data_loader)* batch_size} instances")
        
    return data_loader

if __name__ == '__main__':
    config = [
    {'n_customers': 20, 'max_demand': 10, 'max_distance': 10, 'num_instances': 100}
    # Add more configurations as needed
    ]
    batch_size = 1
    save_to_csv = True
    instance_loader(config, batch_size, save_to_csv)
    