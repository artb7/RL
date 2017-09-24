import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dims):
        super(DQN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.layers = []

        prev_dim = input_size
        for out_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, out_dim))
            self.layers.append(nn.ReLU(inplace=True))
            prev_dim = out_dim

        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out
    
