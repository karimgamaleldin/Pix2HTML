import torch
from torch import nn
from torch.nn import functional as F
from config import CONTEXT_SIZE
from Models.block import VGGBlock
    
class Pix2HTMLV0(nn.Module):
    '''
    Enc and Dec are just names not related to the typical encoder decoder architecture
    '''
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
        
        self.enc = nn.LSTM(19, 128, num_layers=2, batch_first=True)        
        
        self.dec = nn.LSTM(1024 + 128, 512, num_layers=2, batch_first=True)
        self.fc = nn.LazyLinear(vocab_length)
        self.drop = nn.Dropout(p=0.10)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, gui, tokens):
        
        # Extracting features using the image_model
        img = self.image_model(gui)
        
        # Extracting features
        enc, _ = self.enc(tokens)
        
        # Get the output 
        unsqueezed_img = img.unsqueeze(1).repeat(1, enc.shape[1], 1)
        decoder_in = torch.cat([unsqueezed_img, enc], dim=-1)
        dec, _ = self.dec(decoder_in)
        last = dec[:, -1, :]
        fc = self.fc(self.drop(last))
        return fc
        
        