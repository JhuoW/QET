"""Run and evaluation tests run tiny models on shared CPUs: a few threads are much faster than one per core."""

import torch

torch.set_num_threads(min(4, torch.get_num_threads()))
