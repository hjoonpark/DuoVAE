import os
import json
import numpy as np
import torch

def as_np(x):
    return x.detach().cpu().numpy()

def make_directories(root_dir, sub_dirs):
    sub_dirs_out = {}
    for key in sub_dirs:
        sub_dir = os.path.join(root_dir, key)
        os.makedirs(sub_dir, exist_ok=True)
        sub_dirs_out[key] = sub_dir
    return sub_dirs_out