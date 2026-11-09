import torch

# Small models train faster single-threaded than with many competing threads.
torch.set_num_threads(2)
