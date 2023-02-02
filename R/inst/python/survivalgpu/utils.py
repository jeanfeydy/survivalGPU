import time
import torch


def numpy(x):
    return x.detach().cpu().numpy()


def timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# Tensor types:
use_cuda = torch.cuda.is_available()  # Is a GPU available?
device = "cuda" if use_cuda else "cpu"

float32 = torch.float32
int32 = torch.int32
int64 = torch.int64
