from InstanceGenerator import InstanceGenerator

def main(config, filename):
    generator = InstanceGenerator()
    generator.generate_and_save_instances(config, filename)

    print(f"Instances generated based on configuration and saved to respective CSV files.")

instances_config = [
    {'n_customers': 20, 'max_demand': 10, 'max_distance': 20, 'num_instances': 50000}
    # Add more configurations as needed
]

#DEBUG
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\debug_1_2instances.CSV'

#TRAIN
filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_20_50000.CSV'

#VALIDATION
# filename = 

#TEST
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\TSP_test_20_100.CSV'

if __name__ == "__main__":
    main(instances_config, filename)
