import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import *
from torch_geometric_temporal.nn.attention import *
from torch_geometric.nn import GCNConv, CuGraphGATConv
from torch_geometric.nn.models import DeepGCNLayer
'''K stacked GAT layer implementation with pyg https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
if Multihead attention, we concat outputs (default behavior of pyg implementation).
'''
class KLayerGAT(torch.nn.Module): 
    def __init__(self, 
                 K: int, 
                 in_channels: int,  
                 out_channels: int,
                 heads: int, 
                 concat: bool = True,
                 add_self_loops: bool = True): 
        super().__init__()
        self.convs = []
        for k in range(K): 
            if k==0:
                self.convs.append(CuGraphGATConv(in_channels=in_channels,
                                          out_channels=out_channels,
                                          heads=heads,
                                          concat=concat,))
            else: 
                self.convs.append(CuGraphGATConv(in_channels=out_channels,
                                          out_channels=out_channels,
                                          heads=heads,
                                          concat=concat,))
        '''
        Note: saving and loading doesnt work when theres a list like self.convs. 
        So we convert it to nn.sequential. 
        See here: 
            https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/24
        '''
        self.convs = torch.nn.Sequential(*self.convs)
    def forward(self, x, edge_index, edge_weight):
        for k in range(len(self.convs)): 
            x = self.convs[k].forward(x, edge_index, edge_weight)
            return x # shape (N, num_heads * out_channels)


''' 
    Helpful module of K stacked GCN layers
'''
class KLayerGCNConv(torch.nn.Module): 
    def __init__(self, 
                 K: int, 
                 in_channels: int,  
                 out_channels: int,
                 node_dim: int,
                 improved: bool = True,
                 cached: bool = False,
                 add_self_loops: bool = True): 
        super().__init__()
        self.convs = []
        for k in range(K): 
            if k==0:
                self.convs.append(GCNConv(in_channels=in_channels,
                                          out_channels=out_channels,
                                          node_dim=node_dim,
                                          improved=improved,
                                          cached=cached,
                                          add_self_loops=add_self_loops))
            else: 
                self.convs.append(GCNConv(in_channels=out_channels,
                                          out_channels=out_channels,
                                          node_dim=node_dim,
                                          improved=improved,
                                          cached=cached,
                                          add_self_loops=add_self_loops))
        '''
        Note: saving and loading doesnt work when theres a list like self.convs. 
        So we convert it to nn.sequential. 
        See here: 
            https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/24
        '''
        self.convs = torch.nn.Sequential(*self.convs)
    def forward(self, x, edge_index, edge_weight):
        for k in range(len(self.convs)): 
            x = self.convs[k].forward(x, edge_index, edge_weight)
            return x # shape (N, out_channels)
        
'''
    A GCN-GRU model with dense connection, implemented with pyg.DeepGCNLayer 
'''
class DenseGCNGRU(torch.nn.Module):
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
        self.periods = periods # 20
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.densegcn= DeepGCNLayer(conv=KLayerGCNConv(K=3,
                                                       in_channels=self.in_channels,
                                                       out_channels=128,
                                                       node_dim=1,
                                                       improved=self.improved,
                                                       cached=self.cached,
                                                       add_self_loops=self.add_self_loops,
                                                      ),
                                    norm=None,
                                    act=None, #torch.nn.LeakyReLU(),
                                    dropout= 0, #0.1, 
                                    block='dense')
        self.gru = torch.nn.GRU(130,64,2,batch_first=True)
        self.fc = torch.nn.Linear(64, self.periods)
        
    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(X.shape[0],X.shape[1],X.shape[3],130).to(X.device) # (B,N,T_in,F_out_GCN)
        for t in range(X.shape[3]):
            gcn_out = self.densegcn(X[:, :, :, t], edge_index, edge_weight) # (B, N, Fout)
            gru_in[:,:,t,:] = gcn_out
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T_in, F_out_GCN)
        gru_out, _ = self.gru(gru_in) # (B*N,T_in,H)
        out = self.fc(gru_out[:,-1,:]) # (B*N, T_out)
        out = out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,T_out,1)
        return out.squeeze(dim=3) # (B,N,T_out)

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
    
'''
    GCN GRU model without dense connection
'''
class GCNGRU(torch.nn.Module):
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
        self.gcns = KLayerGCNConv( K=3,
                                   in_channels=self.in_channels,
                                   out_channels=128,
                                   node_dim=1,
                                   improved=self.improved,
                                   cached=self.cached,
                                   add_self_loops=self.add_self_loops,
                                  )
#         self.gcn1 = GCNConv(
#             in_channels=self.in_channels,
#             out_channels=128,
#             improved=self.improved,
#             cached=self.cached,
#             add_self_loops=self.add_self_loops,
#         )
#         self.gcn2 = GCNConv(
#             in_channels=128,
#             out_channels=128,
#             improved=self.improved,
#             cached=self.cached,
#             add_self_loops=self.add_self_loops,
#         )
#         self.gcn3 = GCNConv(
#             in_channels=128,
#             out_channels=128,
#             improved=self.improved,
#             cached=self.cached,
#             add_self_loops=self.add_self_loops,
#         )
        self.gru = torch.nn.GRU(128,64,2,batch_first=True)
        self.fc = torch.nn.Linear(64, self.periods)

    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(X.shape[0],X.shape[1],self.periods,128).to(X.device) # (B,N,T,F_out_GCN)
        for period in range(self.periods):
            gcn_out = self.gcns(X[:,:,:,period], edge_index, edge_weight)                 
#             gcn_out = self.gcn1(X[:, :, :, period], edge_index, edge_weight) # (B, N, Fout)
#             gcn_out = self.gcn2(gcn_out, edge_index, edge_weight) # (B, N, Fout)
#             gcn_out = self.gcn3(gcn_out, edge_index, edge_weight) # (B, N, Fout)
            gru_in[:,:,period,:] = gcn_out
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T, F_out_GCN)
        gru_out, _ = self.gru(gru_in) # (B*N,T,H)
        out = self.fc(gru_out[:,-1,:]) # (B*N, Tout)
#         out = F.leaky_relu(out)
        out = out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,Tout,1)
        return out.squeeze(dim=3) # (B,N,T)
    