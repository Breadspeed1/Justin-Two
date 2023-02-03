import torch
import torch.nn as nn

input_size, inner_size, output_size = 386, 500, 128


class Brain:
    input_layer = None
    inner_layer_1 = None
    inner_layer_2 = None
    output_layer = None

    def __init__(self, device="cpu"):
        self.input_layer = torch.randn(input_size, device=device)
        self.inner_layer_1 = torch.randn(inner_size, device=device)
        self.inner_layer_2 = torch.randn(inner_size, device=device)
        self.output_layer = torch.randn(output_size, device=device)

    def calc(self):

