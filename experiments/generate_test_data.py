import os.path as osp
import yaml
from easydict import EasyDict as edict
import argparse
from hgnn.datasets import SyntheticGraphs  # Assuming this is where the dataset class is

def generate_new_test_set(args):
    # Path where the dataset is stored
    dataset_root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/SyntheticGraphs')
    
    # New test node range from args (from the YAML config)
    new_test_node_num = tuple(args.test_node_num)

    # Path to the processed test dataset file
    test_file_path = osp.join(dataset_root, 'processed', 'test.pt')

    # Check if the test set already exists
    if osp.exists(test_file_path):
        print(f"Test set with graphs from {new_test_node_num[0]} to {new_test_node_num[1]} nodes already exists and will be loaded.")
    else:
        print(f"Generating new test set with graphs from {new_test_node_num[0]} to {new_test_node_num[1]} nodes.")
        
        # Define the dataset with the new test node range
        test_dataset = SyntheticGraphs(
            root=dataset_root, 
            split='test',  # Only generate the test set
            train_node_num=(100, 200),  # This won't be used
            test_node_num=new_test_node_num,  # New range for test set
            num_train=0,  # Skip training data generation
            num_val=0,    # Skip validation data generation
            num_test=args.num_test  # Number of test graphs to generate
        )
        
        # Now, the test set has been generated and saved
        print(f"New test set has been generated and saved.")

if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Generate new test set for Synthetic Graphs')
    parser.add_argument('--config', type=str, help='Path to the config file', default='configs/generate_test_set.yaml')
    terminal_args = parser.parse_args()

    # Load the configuration from the YAML file
    with open(terminal_args.config, 'r') as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    # Generate the new test set based on the config
    generate_new_test_set(args)
