import torch
from torch import nn
from torch.nn import functional as F
from config import CONTEXT_SIZE


# class RepeatedVector(nn.Module):
#   def __init__(self, num_repeats):
#     super(RepeatedVector, self).__init__()
#     self.num_repeats = num_repeats

#   def forward(self, x: torch.Tensor):
#     return x.unsqueeze(1).repeat(1, self.num_repeats, 1)

class Pix2CodeV0(nn.Module):

  def __init__(self, input_channels, vocab_length):
    super(Pix2CodeV0, self).__init__()
        
    self.block1 = nn.Sequential(
      nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(p=0.25)
    )

    self.block2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),  
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(p=0.25)
    )

    self.block3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(p=0.25)
    )

    self.block4 = nn.Sequential(
      nn.Flatten(), 
      nn.LazyLinear(1024),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.LazyLinear(1024),
      nn.ReLU(),
      nn.Dropout(p=0.3),
    )


    self.image_model = nn.Sequential(
      *self.block1,
      *self.block2,
      *self.block3,
      *self.block4,
    )

    self.embed = nn.Embedding(vocab_length, 128)
    self.encoder = nn.LSTM(128, 128, 2, batch_first=True)

    self.decoder_lstm = nn.LSTM(1024 + 128, 512, 2, batch_first=True)
    self.out = nn.LazyLinear(vocab_length)
    self.softmax = nn.Softmax(dim=-1)


  def forward(self, gui: torch.Tensor, token: torch.Tensor):
    # Image
    img = self.image_model(gui)
    img_repeated = img.unsqueeze(1).repeat(1, token.size(1), 1)

    # Encoder
    token_embed = self.embed(token)
    token_lstm, _ = self.encoder(token_embed)

    # Decoder
    decoder_in = torch.cat((img_repeated, token_lstm), dim=-1)
    x, _ = self.decoder_lstm(decoder_in)
    x = x[:, -1, :]  # Process the last output for prediction
    x = self.out(x)
    return x