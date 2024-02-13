import torch
from torch import nn
from torch.nn import functional as F
from config import CONTEXT_SIZE

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
    
class Pix2CodeV0(nn.Module):
    def __init__(self, vocab_length):
        super().__init__()
        # Image Model
        image_model_layers = []
        image_model_layers.append(VGGBlock(2, 3, 32, 3))
        image_model_layers.append(VGGBlock(2, 32, 64, 3))
        image_model_layers.append(VGGBlock(2, 64, 128, 3))
        image_model_layers.append(nn.Flatten())
        image_model_layers.append(nn.LazyLinear(1024))
        image_model_layers.append(nn.ReLU(inplace=True))
        image_model_layers.append(nn.Dropout(p=0.3))
        image_model_layers.append(nn.LazyLinear(1024))
        image_model_layers.append(nn.ReLU(inplace=True))
        image_model_layers.append(nn.Dropout(p=0.3))
        self.image_model = nn.Sequential(*image_model_layers)
        
        # Encoder
        self.enc = nn.LSTM(19, 128, num_layers=2, batch_first=True)        
        
        # Decoder
        '''
        At the start we will give the decode 1024 of the image and 128 of the encoder so input is 1024 + 128
        '''
        self.dec = nn.LSTM(1024 + 128, 512, num_layers=2, batch_first=True)
        self.fc = nn.LazyLinear(vocab_length)
        self.drop = nn.Dropout(p=0.10)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, gui, tokens):
        
        # Extracting features using the image_model
        img = self.image_model(gui)
        
        # Extracting features using the encoder
        enc, _ = self.enc(tokens)
        
        # Get the output using the decoder
        unsqueezed_img = img.unsqueeze(1).repeat(1, enc.shape[1], 1)
        decoder_in = torch.cat([unsqueezed_img, enc], dim=-1)
        dec, _ = self.dec(decoder_in)
        last = dec[:, -1, :]
        fc = self.fc(self.drop(last))
        return fc
        