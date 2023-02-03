import torch
import torch.nn as nn
import numpy as np

input_size, inner_size, output_size = 386, 500, 128


class Brain:
    network = None
    device = None

    def __init__(self, device="cpu"):
        self.device = device

        self.network = [
            self.get_layer_layout(input_size, inner_size),
            self.get_layer_layout(inner_size, inner_size),
            self.get_layer_layout(inner_size, output_size),
        ]

    def get_layer_layout(self, in_count, out_count):
        return [
            torch.randn(out_count, in_count, device=self.device),
            torch.randn(out_count, 1, device=self.device)
        ]

    def calc(self, input_tensor):
        data_tensor = input_tensor

        for i in range(0, len(self.network)):
            calc_tensor = torch.zeros_like(self.network[i][1])
            torch.sigmoid(torch.addmm(self.network[i][1], self.network[i][0], data_tensor), out=calc_tensor)
            data_tensor = calc_tensor

        return data_tensor
