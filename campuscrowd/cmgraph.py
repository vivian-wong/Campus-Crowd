import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class CMGraph: 
    def __init__(self, 
                flow_df,
                ADJACENCY_MAT):
        self.flow_df = flow_df
        self.A = torch.from_numpy(ADJACENCY_MAT) 
        self.X = self._get_nodal_matrix()
    
    def _get_nodal_matrix(self):
        """Generates the nodal matrix X by normalizing the number of people and time
        for each PAR and stacking them together.

        `self.X` is expected to be a torch.Tensor of shape 
        (N,D,T) = (num_nodes, num_node_features, num_timesteps)=.

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        
        X_list = []
        common_time = [0, 1e6]

        # find common time shared amongst all PARs
        for par_id, df in self.flow_df.groupby('PAR_id'):
            common_time[0] = max(common_time[0], df['timestamp'].min())
            common_time[1] = min(common_time[1], df['timestamp'].max())
        
        # crop out beginning and end with incomplete data
        for par_id, df in self.flow_df.groupby('PAR_id'):
            df = df.round({'timestamp':1})
            temp_df = df[(common_time[0] <= df['timestamp']) & 
                         (df['timestamp'] <= common_time[1])].copy()
            temp_df['time'] = (temp_df['timestamp'] - common_time[0])/(common_time[1]- common_time[0])
            X_list.append(temp_df[['num_people','time']].to_numpy().transpose([1,0]))
            
        X = np.stack(X_list)
        X = X.astype(np.float32)
        
        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        return torch.from_numpy(X)
            
    def _get_edges_and_weights(self):
        """
        Converts the adjacency matrix to edge indices and edge weights.

        This method uses the dense_to_sparse function from torch_geometric.utils
        to convert the adjacency matrix to edge indices and edge weights, which
        are then stored in the instance variables `self.edges` and `self.edge_weights`.
        """
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in, num_timesteps_out):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target
        
    def get_forecasting_dataset(self, num_timesteps_in=10, num_timesteps_out=10):
        """
        Generates a forecasting dataset for the graph.

        Args:
            num_timesteps_in (int): Number of timesteps the sequence model sees. Default is 10.
            num_timesteps_out (int): Number of timesteps the sequence model has to predict. Default is 10.

        Returns:
            StaticGraphTemporalSignal: A dataset object containing the edges, edge weights, features, and targets.

        Note:
            One step does not necessarily mean 1 second. The actual time depends on your data's timestamp column.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset    