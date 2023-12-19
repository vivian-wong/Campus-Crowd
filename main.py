import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2, glob, os, torch, math, torchvision, datetime
from tqdm import tqdm 
import torch.nn.functional as F
import configparser
import argparse

# my files
from cmgraph import parse_gcs, image_to_world, GCSDatasetLoaderStatic, DatasetLoaderStatic
from models import DenseGCNGRU, GRU_only, GCNGRU, A3TGCN_2, TGCN_2

'''modified from notebook for reporting experimental results. 
note: environmental conflict on GAT hasn't been resolved. so we've only doing GCN GRU encoders on the 3 datasets
'''
def get_configs_dict(cfg_file_path): 
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(cfg_file_path)

    # Store the settings in a dictionary
    configs_dict = {}

    # Iterate over sections
    for section in config.sections():
        # Create a nested dictionary for each section
        configs_dict[section] = {}
        # Iterate over options in the section
        for option in config.options(section):
            # Get the value of the option
            value = config.get(section, option)
            # Check if the value is a string representation of a NumPy array
            if ',' in value:
                # Convert the string to a 1D NumPy array
                arr = value.split('\n')
                # Split the values by commas and convert each row to a NumPy array
                arr = np.array([row.split(',') for row in arr], dtype=float)
                # Store the NumPy array in the nested dictionary
                configs_dict[section][option] = arr
            else:
                # Store the value as-is in the nested dictionary
                configs_dict[section][option] = value
    return configs_dict

def get_pyg_temporal_dataset(DATASET, forecasting_horizon): 
    '''returns: pytorch geometric temporal static graph temporal signal object'''
    if DATASET == 'GCS':
        trajs= parse_gcs("GCS/Annotation") # 35 seconds to load
        ZONE_LIST =[
            (0, 0, 28, 14),
            (28, 0, 55, 14),
            (0, 14, 28, 24),
            (28, 14, 55, 24),
            (0, 24, 28, 35),
            (28, 24, 55, 35),
            (0, 35, 28, 45),
            (28, 35, 55, 45),
            (0, 45, 55, 55)
        ]
        loader = GCSDatasetLoaderStatic(
            trajs=trajs,
            ZONE_LIST = ZONE_LIST)
    if DATASET == 'SEQ': 
        configs_dict = get_configs_dict('SEQ.cfg')
        flow_df = pd.read_csv('flow_df_SEQ_1fps.csv')
        all_zone_dfs = [g for _,g in flow_df.groupby('egress_region')]
        loader = DatasetLoaderStatic(all_zone_dfs,
                                     configs_dict['CMGraph']['adjacency_mat'])
    if DATASET == 'STADIUM_2023': 
        configs_dict = get_configs_dict('Stadium_2023.cfg')
        flow_df = pd.read_csv('flow_df_stadium_2023_1fps.csv')
        all_zone_dfs = [g for _,g in flow_df.groupby('egress_region')]
        loader = DatasetLoaderStatic(all_zone_dfs,
                                     configs_dict['CMGraph']['adjacency_mat'])
        
    dataset = loader.get_dataset(num_timesteps_in=forecasting_horizon, 
                             num_timesteps_out=forecasting_horizon)
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ",  len(list(dataset)))
    print(next(iter(dataset))) # Show first sample
    return dataset

def get_loaders(dataset, batch_size, train_ratio, val_ratio, test_ratio, device, manual_seed=42): 
    # Check if the ratios sum up to 1
    total_ratio = train_ratio + test_ratio + val_ratio
    if not (total_ratio == 1.0):
        raise ValueError("Ratios must sum up to 1.0. Please provide valid ratios.")
    
    # convert node features to tensor dataset
    input_np = np.array(dataset.features) 
    target_np = np.array(dataset.targets) 
    input_tensor = torch.from_numpy(input_np).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    target_tensor = torch.from_numpy(target_np).type(torch.FloatTensor).to(device)  # (B, N, T)
    dataset_new = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    
    # split to train val test and get loader
    lengths = [int(p * len(dataset_new)) for p in [train_ratio, val_ratio, test_ratio]]
    lengths[-1] = len(dataset_new) - sum(lengths[:-1])

    train_dataset_new, val_dataset_new, test_dataset_new = torch.utils.data.random_split(
        dataset_new, 
        lengths, 
        generator=torch.Generator().manual_seed(manual_seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset_new, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=False)

    print("Number of train buckets: ", len(list(train_dataset_new)))
    print("Number of val buckets: ", len(list(val_dataset_new)))
    print("Number of test buckets: ", len(list(test_dataset_new)))
    return train_loader, val_loader, test_loader

def training_loop(model, train_loader, val_loader, static_edge_index, num_epochs, lr):
    loss_fn = torch.nn.MSELoss() 
    train_epoch_losses = []
    val_epoch_losses = []
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for snapshot in dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break;

    # start training loop 
    for epoch in range(num_epochs):
        step = 0
        loss_list = []
        for encoder_inputs, labels in train_loader:
            y_hat = model(encoder_inputs, static_edge_index)         # Get model predictions
            loss = loss_fn(y_hat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if step % 100 == 0 :
                print("Epoch {} Step {} train MSE: {:.4f}".format(epoch, step, sum(loss_list)/len(loss_list)))
            step= step+ 1
            
        avg_train_loss = sum(loss_list) / len(loss_list)
        train_epoch_losses.append(avg_train_loss)
        print("Epoch {} Average Training MSE: {:.4f}".format(epoch, avg_train_loss))
    
        # Validation
        if len(val_loader) > 0: 
            model.eval()
            val_loss_list = []
            with torch.no_grad():
                for val_encoder_inputs, val_labels in val_loader:
                    val_y_hat = model(val_encoder_inputs, static_edge_index)
                    val_loss = loss_fn(val_y_hat, val_labels)
                    val_loss_list.append(val_loss.item())

            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            val_epoch_losses.append(avg_val_loss)
            print("Epoch {} Average Validation MSE: {:.4f}".format(epoch, avg_val_loss))
            model.train()
        
    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr': lr,
                        'train_epoch_losses': train_epoch_losses,
                        'val_epoch_losses': val_epoch_losses,
                      }
    return model, checkpoint_dict

def testing_loop(model, test_loader, static_edge_index, checkpoint_dict=dict()): 
    loss_fn = torch.nn.MSELoss()
    torch.no_grad()
    model.eval()
    # Store for analysis
    total_loss = []
    total_mae = []
    
    for encoder_inputs, labels in test_loader:
        # Get model predictions
        y_hat = model(encoder_inputs, static_edge_index)
        # Mean squared error
        loss = loss_fn(y_hat, labels)
        mae = F.l1_loss(y_hat, labels)
        total_loss.append(loss.item())
        total_mae.append(mae.item())
        
    # update checkpoint_dict
    checkpoint_dict['test_mse'] = sum(total_loss)/len(total_loss)
    checkpoint_dict['test_mae'] = sum(total_mae)/len(total_mae)
    
    print("Test MSE: {:.4f}".format(checkpoint_dict['test_mse']))
    print("Test MAE: {:.4f}".format(checkpoint_dict['test_mae']))
    
    return model, checkpoint_dict

def save_or_update_checkpoint(checkpoint_dict, path):
    # save trained model
    torch.save(checkpoint_dict, path)
    print('Model/results saved to '+path)
    
def load_checkpoint(model, path): 
    checkpoint_dict = torch.load(path)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint_dict['lr'])
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    print(model)
    return model, optimizer, checkpoint_dict

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Machine Learning Experiment Template")

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
    
    ###### NOTE TO YIRONG: THIS SECTION IS WHERE I LOAD DATASET.
    # get pytorch dataloaders
    dataset = get_pyg_temporal_dataset(args.DATASET, args.forecasting_horizon)
    train_loader, val_loader, test_loader = get_loaders(dataset, 
                                                        args.batch_size, 
                                                        args.train_ratio, 
                                                        args.val_ratio, 
                                                        args.test_ratio, 
                                                        device)
    ###### NOTE TO YIRONG: ANYTHING BELOW THIS LINE IS PROBABLY NOT HELPFUL TO YOU. 
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
    model, checkpoint_dict = training_loop( model, 
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
    model, checkpoint_dict = testing_loop(model, test_loader, static_edge_index, checkpoint_dict=checkpoint_dict)
    # update save to include test mse and mae. 
    if args.save_model:
        save_or_update_checkpoint(checkpoint_dict, path)