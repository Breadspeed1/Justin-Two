import torch
import torch.nn as nn
import numpy as np
import Brain

print(f"using cuda: {torch.cuda.is_available()}")

device = "cuda"

brain = Brain.Brain(device=device)
print(brain.calc(torch.randn(386, 1, device=device)))
