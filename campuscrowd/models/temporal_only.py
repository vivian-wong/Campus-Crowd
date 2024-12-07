import torch
from torch_geometric_temporal.nn.recurrent import *

'''
    Simple GRU model (does not use edge_index)
'''
class GRU_only(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.gru = torch.nn.GRU(self.in_channels,64,2,batch_first=True)
        self.fc = torch.nn.Linear(64, self.periods)

    def forward( self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor = None,  # dummy placeholder
                edge_weight: torch.FloatTensor = None, # dummy placeholder
               ) -> torch.FloatTensor:
        gru_in = torch.reshape(X, (X.shape[0], X.shape[1], self.periods, -1)) #(B,N,2,T)->(B,N,T,2)
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T, 2)        
        gru_out, _ = self.gru(gru_in) # (B*N,T_in,H)
        out = self.fc(gru_out[:,-1,:]) # (B*N, T_out)
        out = out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,T_out,1)
        return out.squeeze(dim=3) # (B,N,T_out)
    