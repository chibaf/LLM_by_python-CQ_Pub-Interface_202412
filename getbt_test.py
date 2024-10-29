# get batch test
import torch
from getbt import get_batch
#
xb, yb = get_batch('train')
print(xb)
print(yb)