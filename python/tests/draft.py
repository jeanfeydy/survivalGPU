import numpy as np
import torch


def torch_lexsort(a, dim=-1):
    assert dim == -1  # Transpose if you want differently
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
    # To be consistent with numpy, we flip the keys (sort by last row first)
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)


# Make random float vector with duplicates to test if it handles floating point well
vals = torch.arange(2)
a = vals[(torch.rand(3, 9) * 2).long()]

print(a)
ind = torch_lexsort(a)
ind_np = torch.from_numpy(np.lexsort(a.numpy()))
print("Torch ind", ind)
print("Numpy ind", ind)

print("Torch result")
print(a[:, ind])
print("Numpy result")
print(a[:, ind_np])
