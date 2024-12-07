# import all the necessary libraries
import os
import sys
import torch
import argparse

# for relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from campuscrowd.data_utils import get_pyg_temporal_dataset, get_loaders
from campuscrowd.models import DenseGCNGRU, GCNGRU, GRU_only
from campuscrowd import train, test, save_or_update_checkpoint

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Campus Crowd Forecasting Experiment")

    # Add arguments for your experiment
    parser.add_argument("--DATASET", type=str, help="Name of dataset. Available options: 'GCS', 'SEQ', 'STADIUM_2023'")
    parser.add_argument("--MODEL", type=str, help="Name of model. Available options: 'GRU', 'GCNGRU', 'DenseGCNGRU'")
    parser.add_argument("--forecasting_horizon", type=int, default=20, help="Number of steps for input and for forecasting.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio for training set (e.g., 0.8 for 80%)")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Ratio for testing set (e.g., 0.2 for 20%)")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Ratio for validation set (e.g., 0.1 for 10%)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for train+val+test")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=40, help="Total epochs for training")
    parser.add_argument("--save_model", type=bool, default=False, help="Save model state dict and losses in args.save_dir as a dictionary (see checkpoint_dict)")
    parser.add_argument("--save_dir", type=str, default='./checkpoints', help="Folder to save model state dicts and losses. Only useful when args.save_model is True")

    # Parse command-line arguments
    args = parser.parse_args()
    assert args.DATASET in ['GCS', 'SEQ', 'STADIUM_2023']
    assert args.MODEL in ['GRU', 'GCNGRU', 'DenseGCNGRU']
    
    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # get pytorch dataloaders
    dataset = get_pyg_temporal_dataset(args.DATASET, args.forecasting_horizon)
    train_loader, val_loader, test_loader = get_loaders(dataset, 
                                                        args.batch_size, 
                                                        args.train_ratio, 
                                                        args.val_ratio, 
                                                        args.test_ratio, 
                                                        device)
    # get static edge index (i.e. adjacency matrix). Only need to do this one since edge index doesn't change for each CMGraph. 
    for snapshot in dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break;
        
    # build model 
    if args.MODEL == 'GRU':
        model = GRU_only(in_channels=2, 
                         periods=args.forecasting_horizon, 
                         batch_size=args.batch_size).to(device)
    elif args.MODEL == 'GCNGRU':
        model = GCNGRU(in_channels=2, 
                       periods=args.forecasting_horizon, 
                       batch_size=args.batch_size).to(device)
    elif args.MODEL == 'DenseGCNGRU': 
        model = DenseGCNGRU(in_channels=2, 
                            periods=args.forecasting_horizon, 
                            batch_size=args.batch_size).to(device) 
    
    # train model
    model, checkpoint_dict = train( model, 
                                    train_loader, 
                                    val_loader, 
                                    static_edge_index, 
                                    num_epochs=args.epochs, lr=args.lr
                                    )
    if args.save_model:
        filename = model.__class__.__name__+'_'+args.DATASET+'_'+'{}_steps'.format(args.forecasting_horizon)+'.pt'
        path = os.path.join(args.save_dir,
                            filename)
        save_or_update_checkpoint(checkpoint_dict, path)
    
    # test model
    model, checkpoint_dict = evaluate(model, test_loader, static_edge_index, checkpoint_dict=checkpoint_dict)
    # update save to include test mse and mae. 
    if args.save_model:
        save_or_update_checkpoint(checkpoint_dict, path)