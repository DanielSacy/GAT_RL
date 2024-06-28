from InstanceGenerator import InstanceGenerator

def main(config, filename):
    generator = InstanceGenerator()
    generator.generate_and_save_instances(config, filename)

    print(f"Instances generated based on configuration and saved to respective CSV files.")

instances_config = [
    {'n_customers': 10, 'max_demand': 200, 'max_distance': 100, 'num_instances': 10}
    # {'n_customers': 10, 'max_demand': 200, 'max_distance': 100, 'num_instances': 10000},
    # Add more configurations as needed
]

filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instance_creator\instance_data.csv'

if __name__ == "__main__":
    main(instances_config, filename)
