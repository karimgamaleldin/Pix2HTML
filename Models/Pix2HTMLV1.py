import torch
from torch import nn
from torch.nn import functional as F
from config import CONTEXT_SIZE
from Models.block import VGGBlock, PositionalEncoding

class Pix2HTMLV1(nn.Module):
    def __init__(self, vocab_length, d_model=512, nhead=8, num_decoder_layers=4, dim_feedforward=2048, ):
        super().__init__()
        self.vocab_length = vocab_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
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
        
        # Image and Token projection layers
        self.token_projection = nn.Linear(19, d_model)
        self.pos_enc = PositionalEncoding()
        self.image_projection = nn.LazyLinear(d_model)
        
        # decoder
        dec_lay = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_lay, num_layers=num_decoder_layers)
        
        # out
        self.fc_out = nn.Linear(d_model, vocab_length)
    def forward(self, gui, tokens):
        
        # Extracting features using the image_model
        img = self.image_model(gui)
        
        # Projecting the image_features and the tokens
        tokens_proj = self.pos_enc(self.token_projection(tokens.view(-1, self.vocab_length)).view(-1, 48, self.d_model))
        memory = self.image_projection(img).unsqueeze(1).repeat(1, tokens.shape[1] , 1)
        mask = self.generate_square_subsequent_mask(tokens.shape[1])
        
        tokens_dec = self.transformer_decoder(tokens_proj, memory, tgt_mask=mask)
        
        fc = self.fc_out(tokens_dec[:, -1, :])
        
        return fc
    
    def generate_square_subsequent_mask(self, seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
        return mask
        