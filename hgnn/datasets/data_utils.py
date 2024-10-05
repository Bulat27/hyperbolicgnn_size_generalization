import os.path as osp
import numpy as np
import random
from torch_geometric.datasets import TUDataset
from torch.utils.data import Subset


def get_split_indices(dataset_root, num_total, num_train, num_val, num_test):
    """
    Checks if the split files exist. If they don't, it creates and saves random splits for train/val/test.
    
    Args:
        dataset_root (str): Root directory where the dataset is located.
        num_total (int): Total number of graphs available (e.g., 1113 for the PROTEINS dataset).
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        num_test (int): Number of test samples.

    Returns:
        Tuple[List[int], List[int], List[int]]: Indices for train, validation, and test splits.
    """
    train_idx_path = osp.join(dataset_root, 'train_idx.txt')
    val_idx_path = osp.join(dataset_root, 'val_idx.txt')
    test_idx_path = osp.join(dataset_root, 'test_idx.txt')

    # If all split files exist, load the predefined splits
    if osp.exists(train_idx_path) and osp.exists(val_idx_path) and osp.exists(test_idx_path):
        train_idx = np.loadtxt(train_idx_path, dtype=np.int64)
        val_idx = np.loadtxt(val_idx_path, dtype=np.int64)
        test_idx = np.loadtxt(test_idx_path, dtype=np.int64)
    else:
        # Assuming num_total is the total number of graphs available (e.g., 1113)
        # Generate random indices from the total pool of graphs
        total_indices = list(range(num_total))
        random.shuffle(total_indices)  # Shuffle to ensure randomness
        
        # Take the first `num_train + num_val + num_test` graphs from the shuffled total
        selected_indices = total_indices[:num_train + num_val + num_test]

        # Further shuffle selected_indices to ensure randomness within the subset
        random.shuffle(selected_indices)

        # Split selected graphs into train, validation, and test sets
        train_idx = selected_indices[:num_train]
        val_idx = selected_indices[num_train:num_train + num_val]
        test_idx = selected_indices[num_train + num_val:num_train + num_val + num_test]

        # Save the splits
        np.savetxt(train_idx_path, train_idx, fmt='%d')
        np.savetxt(val_idx_path, val_idx, fmt='%d')
        np.savetxt(test_idx_path, test_idx, fmt='%d')

    return train_idx, val_idx, test_idx

def split_tudataset(dataset_root, num_total, num_train, num_val, num_test):
    """
    Splits the TUDataset based on saved or newly generated indices for train/validation.
    
    Args:
        dataset_root (str): Root directory where the dataset is located.
        num_total (int): Total number of available graphs.
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        num_test (int): Number of test samples.

    Returns:
        Tuple[Subset, Subset]: Train and validation subsets.
    """
    dataset = TUDataset(dataset_root, name='PROTEINS')  # Change as needed

    # Get the split indices (either loaded or newly generated)
    train_idx, val_idx, _ = get_split_indices(dataset_root, num_total, num_train, num_val, num_test)

    # Create the subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset


def load_test_tudataset(dataset_root):
    """
    Loads the test dataset based on saved indices. 
    If the test split file does not exist, it raises an error.

    Args:
        dataset_root (str): Root directory where the dataset is located.

    Returns:
        Subset: Test dataset.

    Raises:
        FileNotFoundError: If the test split file does not exist.
    """
    dataset = TUDataset(dataset_root, name='PROTEINS')  # Change as needed

    # Path to the test split file
    test_idx_path = osp.join(dataset_root, 'test_idx.txt')

    # Check if the test split file exists
    if not osp.exists(test_idx_path):
        raise FileNotFoundError(f"Test split file not found at {test_idx_path}. Please provide a valid test split.")

    # Load the test split indices
    test_idx = np.loadtxt(test_idx_path, dtype=np.int64)

    # Create the test subset
    test_dataset = Subset(dataset, test_idx)

    return test_dataset
