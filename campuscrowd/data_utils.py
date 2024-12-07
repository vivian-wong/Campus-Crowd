import pandas as pd
import numpy as np
import configparser
import torch
from campuscrowd.cmgraph import CMGraph

config_file_paths = {
    'SEQ':'./campuscrowd/data_processed/SEQ.cfg', 
    'Stadium':'./campuscrowd/data_processed/Stadium_2023.cfg'
}

def get_pyg_temporal_dataset(DATASET, forecasting_horizon): 
    '''
    Parameters:
        raise ValueError(f"DATASET must be one of: {', '.join(config_file_paths.keys())}")
    forecasting_horizon (int): The number of timesteps for forecasting.

    Returns:
    torch_geometric_temporal.signal.StaticGraphTemporalSignal: 
        A static graph temporal signal object containing the dataset.
    '''

    if DATASET not in config_file_paths.keys():
        raise ValueError("DATASET must be one of: ", config_file_paths.keys())
    
    configs_dict = get_configs_dict(config_file_paths[DATASET])
    flow_df = pd.read_csv(configs_dict['csv-path']['flow_df_path'])
    cmgraph = CMGraph(flow_df, configs_dict['CMGraph']['adjacency_mat'])
    dataset = cmgraph.get_forecasting_dataset(
        num_timesteps_in =forecasting_horizon, 
        num_timesteps_out=forecasting_horizon
    )
    
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ",  len(list(dataset)))
    print(next(iter(dataset))) # Print the first sample of the dataset
    return dataset, cmgraph

def get_loaders(dataset, batch_size, train_ratio, val_ratio, test_ratio, device, manual_seed=42): 
    """
    Splits the dataset into training, validation, and test sets, and returns data loaders for each set.
    Args:
        dataset (Dataset): The dataset containing features and targets.
        batch_size (int): The number of samples per batch to load.
        train_ratio (float): The ratio of the dataset to be used for training [0-1].
        val_ratio (float): The ratio of the dataset to be used for validation [0-1].
        test_ratio (float): The ratio of the dataset to be used for testing [0-1].
        device (torch.device): The device on which to load the tensors (e.g., 'cpu' or 'cuda').
        manual_seed (int, optional): The seed for random number generation. Default is 42.
    Returns:
        tuple: A tuple containing three DataLoader objects for the training, validation, and test sets.
    Raises:
        ValueError: If the sum of train_ratio, val_ratio, and test_ratio is not equal to 1.0.
    """
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

def get_configs_dict(cfg_file_path): 
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read_file(open(cfg_file_path))

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

