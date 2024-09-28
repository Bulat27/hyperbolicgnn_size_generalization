import os, os.path as osp
import time
import warnings
from progressbar import progressbar

import yaml
from easydict import EasyDict as edict
import argparse
import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from hgnn.datasets import SyntheticGraphs
from hgnn.models import GraphClassification
from hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold

def train(args):
    # Create and load datasets
    dataset_root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/SyntheticGraphs')
    transform = T.Compose((
        T.ToUndirected(),
        T.OneHotDegree(args.in_features - 1, cat=False)
    ))
    train_dataset = SyntheticGraphs(dataset_root, split='train', transform=transform, node_num=(args.node_num_min, args.node_num_max), num_train=args.num_train, num_val=args.num_val,  num_test=args.num_test)
    val_dataset = SyntheticGraphs(dataset_root, split='val', transform=transform, node_num=(args.node_num_min, args.node_num_max), num_train=args.num_train, num_val=args.num_val, num_test=args.num_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

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

    # Setup model
    model = GraphClassification(args, manifold).to(args.device)

    # And optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.optimizer == 'amsgrad', weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    # Train, store model with best accuracy on validation set
    best_accuracy = 0
    for epoch in progressbar(range(args.epochs), redirect_stdout=True):
        model.train()

        total_loss = 0
        for data in train_loader:
            model.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = loss_function(out, data.y)
            loss.backward(retain_graph=True)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        val_acc = evaluate(args, model, val_loader)
        train_loss = total_loss / len(train_loader)
        if args.verbose:
            print('Epoch {:n} - training loss {:.3f}, validation accuracy {:.3f}'.format(epoch, train_loss, val_acc))
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), osp.join(args.logdir, 'best.pt'))
            best_accuracy = val_acc
        args.writer.add_scalar('training loss', train_loss, epoch)
        args.writer.add_scalar('validation accuracy', val_acc, epoch)


def evaluate(args, model, data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        data = data.to(args.device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)


if __name__ == "__main__":
    file_dir = osp.dirname(osp.realpath(__file__))

    # Parse arguments from command line
    parser = argparse.ArgumentParser('Synthetic Graph classification with Hyperbolic GNNs')
    parser.add_argument('--config', type=str, default=osp.join(file_dir, 'configs/synth_euclidean.yaml'), help='config file')
    parser.add_argument('--embed_dim', type=int, help='dimension for embedding')
    parser.add_argument('--log_timestamp', type=str, help='timestamp used to name the log directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='print intermediate scores')
    terminal_args = parser.parse_args()

    # Parse arguments from config file
    with open(terminal_args.config) as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    # Additional arguments
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.verbose = terminal_args.verbose
    if terminal_args.embed_dim is not None:
        args.embed_dim = terminal_args.embed_dim

    # Create log directory
    experiment_name = 'hgnn_{}_dim{}'.format(args.manifold, args.embed_dim)
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))
    if terminal_args.log_timestamp is not None:
        run_time = terminal_args.log_timestamp
    args.logdir = osp.join(file_dir, 'logs', experiment_name, run_time)
    if not osp.exists(args.logdir):
        os.makedirs(args.logdir)

    # Setup tensorboard writer
    args.writer = SummaryWriter(args.logdir)

    # Set seeds
    random.seed(terminal_args.seed)
    np.random.seed(terminal_args.seed)
    torch.manual_seed(terminal_args.seed)
    torch.cuda.manual_seed(terminal_args.seed)
    torch.cuda.manual_seed_all(terminal_args.seed)

    print('Training {}'.format(experiment_name))
    train(args)