import torch
import torch.nn as nn

cuda0 = torch.device('cuda:0')
print(f"using cuda: {torch.cuda.is_available()}")

