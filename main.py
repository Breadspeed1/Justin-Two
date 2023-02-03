import torch
import torch.nn as nn
import numpy as np
import Brain

print(f"using cuda: {torch.cuda.is_available()}")

brain = Brain.Brain()
print(brain.calc(torch.randn(386, 1)))
