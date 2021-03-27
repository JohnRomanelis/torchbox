import torch
import torch.nn as nn 

class ReLU_fai(nn.Module):

    def forward(self, x):
        return torch.clamp(x, 0) - 0.5