import math
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, kernel_size):
        super(VGGBlock, self).__init__() 
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout(p=0.25))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x) 
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=48):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe
        return x
    