import torch
import os

TORCH = torch.__version__.split("+")[0]
if not torch.cuda.is_available():
    print(f"No CUDA found for torch=={TORCH}.")
    CUDA='cpu'
else:
    CUDA = "cu" + torch.version.cuda.replace(".","")
print(f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html")
os.system(f'pip install torch-scatter torch-sparse     -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html')
os.system('pip install torch-geometric==2.3')
os.system('pip install torch-geometric-temporal')
os.system('pip install pandas==2.2.3') # temporary solution for pandas, numpy and pyg-temporal version conflict
import torch_geometric
import torch_geometric_temporal

