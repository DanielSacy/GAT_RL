from InstanceGenerator import InstanceGenerator

def main(config, filename):
    generator = InstanceGenerator()
    generator.generate_and_save_instances(config, filename)

    print(f"Instances generated based on configuration and saved to respective CSV files.")

instances_config = [
    {'n_customers': 3, 'max_demand': 200, 'max_distance': 100, 'num_instances': 20}
    # {'n_customers': 10, 'max_demand': 200, 'max_distance': 100, 'num_instances': 1000}
    # {'n_customers': 10, 'max_demand': 200, 'max_distance': 100, 'num_instances': 10000},
    # Add more configurations as needed
]

#TRAIN
filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_20.CSV'
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_100.CSV'
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_1000.CSV'
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\train\train_10000.CSV'

#VALIDATION
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\validation\val_100.CSV'
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\validation\val_1000.CSV'

#TEST
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\test_100.CSV'
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\test_1000.CSV'
# filename = r'D:\DAY2DAY\MESTRADO\Codes\GNN\GAT_VRP1\gat_vrp1\src_batch\instances\test\test_10000.CSV'


if __name__ == "__main__":
    main(instances_config, filename)
