import json
import torch
import torch.nn as nn
import torch.nn.functional as F
#setting of hyper parameters
batch_size=16
block_size=500
max_iters=50000000
eval_interval=444
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'

eval_iters=444
n_embd=64
n_head=4
n_layer=4
dropout=0.0
torch.manual_seed(1337)

print("check ended.")