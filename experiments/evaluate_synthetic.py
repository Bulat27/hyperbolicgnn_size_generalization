import os, os.path as osp
import time
import warnings

import yaml
from easydict import EasyDict as edict
import argparse
import numpy as np
import random

import torch
from sklearn.metrics import f1_score, matthews_corrcoef

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from hgnn.datasets import SyntheticGraphs
from hgnn.models import GraphClassification
from hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold

from hgnn.datasets.data_utils import load_test_tudataset

def test(args):

    # Load test dataset for SyntheticGraphs or TUDataset
    
    if args.dataset == 'synthetic':
        dataset_root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/SyntheticGraphs')
        transform = T.Compose((
            T.ToUndirected(),
            T.OneHotDegree(args.in_features - 1, cat=False)
        ))

        test_dataset = SyntheticGraphs(dataset_root, split='test', transform=transform, train_node_num=tuple(args.train_node_num), test_node_num=tuple(args.test_node_num), num_train=args.num_train, num_val=args.num_val, num_test=args.num_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    else:
        # TUDataset case: load the test dataset and raise an error if the split doesn't exist
        dataset_root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/PROTEINS') # Can be abstracted to any TUDataset!

        test_dataset = load_test_tudataset(dataset_root=dataset_root)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Select manifold
    if args.manifold == 'euclidean':
        manifold = EuclideanManifold()
    elif args.manifold == 'poincare':
        manifold = PoincareBallManifold()
    elif args.manifold == 'lorentz':
        manifold = LorentzManifold()
        args.embed_dim += 1
    else:
        manifold = EuclideanManifold()
        warnings.warn('No valid manifold was given as input, using Euclidean as default')

    # Setup model and load checkpoint
    model = GraphClassification(args, manifold).to(args.device)
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

    # Evaluate on test set and store results in csv if specified
    test_acc, test_f1, test_mcc = evaluate(args, model, test_loader)
    print('Test accuracy {:.4f}, f1 score {:.4f} mcc {:.4f}'.format(test_acc, test_f1, test_mcc))
    if args.csv_file is not None:
        with open(args.csv_file, "a") as f:
            f.write('{}, {}, {:.4f}, {:.4f}, {:.4f}\n'.format(args.manifold, args.embed_dim - 1 * (args.manifold == 'lorentz'), test_acc, test_f1, test_mcc))

def evaluate(args, model, data_loader):
    model.eval()
    correct = 0
    pred_list = []
    true_list = []
    inference_time = 0

    for data in data_loader:
        data = data.to(args.device)
        with torch.no_grad():
            start_time = time.time()  # Start timer for forward pass
            pred = model(data).max(dim=1)[1]
            inference_time += time.time() - start_time  # Accumulate inference time
        
        correct += pred.eq(data.y).sum().item()
        pred_list.append(pred.cpu().numpy())
        true_list.append(data.y.cpu().numpy())
    
    # Convert the lists to numpy arrays
    y_true = np.concatenate(true_list)
    y_pred = np.concatenate(pred_list)
    
    # Calculate accuracy, F1, and MCC
    accuracy = correct / len(data_loader.dataset)
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Inference time: {inference_time:.4f} seconds")  # Print total inference time
    return accuracy, f1, mcc

if __name__ == "__main__":
    file_dir = osp.dirname(osp.realpath(__file__))

    # Parse arguments from command line
    parser = argparse.ArgumentParser('Synthetic Graph classification with Hyperbolic GNNs')
    parser.add_argument('--config', type=str, default=osp.join(file_dir, 'configs/synth_euclidean.yaml'), help='config file')
    parser.add_argument('--embed_dim', type=int, help='dimension for embedding')
    parser.add_argument('--log_timestamp', type=str, help='timestamp used for the log directory where the checkpoint is located')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--csv_file', type=str, help='csv file to output results')
    parser.add_argument('--dataset', type=str, help='Dataset name (synthetic or PROTEINS)')
    terminal_args = parser.parse_args()

    # Parse arguments from config file
    with open(terminal_args.config) as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    # Additional arguments
    if terminal_args.embed_dim is not None:
        args.embed_dim = terminal_args.embed_dim
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    experiment_name = 'hgnn_{}_dim{}'.format(args.manifold, args.embed_dim)
    args.checkpoint = osp.join(file_dir, 'logs', experiment_name, terminal_args.log_timestamp, 'best.pt')
    args.dataset = terminal_args.dataset

    # Create csv file for output
    args.csv_file = None
    if terminal_args.csv_file is not None:
        output_dir = osp.join(file_dir, 'output')
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        args.csv_file = osp.join(output_dir, terminal_args.csv_file)
        if not osp.isfile(args.csv_file):
            with open(args.csv_file, "w") as f:
                f.write('manifold, dimensions, accuracy, f1, mcc\n')

    # Manual seed
    random.seed(terminal_args.seed)
    np.random.seed(terminal_args.seed)
    torch.manual_seed(terminal_args.seed)
    torch.cuda.manual_seed(terminal_args.seed)
    torch.cuda.manual_seed_all(terminal_args.seed)

    print('Evaluating {}'.format(experiment_name))
    test(args)