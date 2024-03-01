import torch
from torch import nn
from torch.nn import functional as F
from config import CONTEXT_SIZE
from Models.block import VGGBlock
import pytorch_lightning as pl
from torch import optim
import torchmetrics
    
class Pix2HTMLV0(pl.LightningModule):
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
        
        # metriccs
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    def forward(self, gui, seq):
        # Extracting features using the image_model
        img = self.image_model(gui)
        # Extracting features
        enc, _ = self.enc(seq)
        # Get the output 
        unsqueezed_img = img.unsqueeze(1).repeat(1, enc.shape[1], 1)
        decoder_in = torch.cat([unsqueezed_img, enc], dim=-1)
        dec, _ = self.dec(decoder_in)
        last = dec[:, -1, :]
        fc = self.fc(self.drop(last))
        return fc
    
    def training_step(self, batch, batch_idx):
        loss, logits, targets = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, targets)
        self.log_dict({'train_loss': loss, 'train_accuracy:': accuracy}, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'logits': logits, 'targets': targets}
    
    def training_epoch_end(self, outputs):
        pass 
    
    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, targets)
        self.log_dict({'train_loss': loss, 'train_accuracy:': accuracy}, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'logits': logits, 'targets': targets}
    
    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, targets)
        self.log_dict({'train_loss': loss, 'train_accuracy:': accuracy}, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'logits': logits, 'targets': targets}
    
    def _common_step(self, batch, batch_idx):
        # Get the data
        gui = batch['gui']
        seq = batch['seq']
        targets = batch['targets']
        # Forward pass
        logits = self.forward(gui, seq)
        # Calculating the loss
        loss = F.cross_entropy(logits, targets)
        return loss, logits, targets
    
    def predict_step(self, batch, batch_idx):
        gui = batch['gui']
        seq = batch['seq']
        # Forward pass
        logits = self.forward(gui, seq)
        # Getting the argmax
        pred = logits.argmax(1)
        return pred
        
    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=1e-4)