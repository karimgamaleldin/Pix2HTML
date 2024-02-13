from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from config import BATCH_SIZE

# Dataset
class Pix2CodeDataset(Dataset):
    def __init__(self, tokens, labels, gui, txt):
        self.tokens = tokens.reshape(-1, 48, 19)
        self.labels = labels.reshape(-1, 19)
        self.gui = gui
        self.txt = txt
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        token = torch.tensor(self.tokens[index]).float()
        label = torch.tensor(self.labels[index]).float()
        gui = torch.tensor(self.gui[index // 42]).float()
        txt = self.txt[index // 42]
        
        return {
            'token': token,
            'label': label,
            'gui': gui,
            'txt': txt
        }


def create_dataloaders(train_tokens, train_labels, train_gui, train_txt, test_tokens, test_labels, test_gui, test_txt):
    train_dataset = Pix2CodeDataset(train_tokens, train_labels, train_gui, train_txt)
    test_dataset = Pix2CodeDataset(test_tokens, test_labels, test_gui, test_txt)
    def pix2code_collate_fn(batch):
        tokens = [item['token'] for item in batch]
        labels = [item['label'] for item in batch]
        guis = [item['gui'] for item in batch]
        txt_batch = [item['txt'] for item in batch]
        
        # stack them in a single tensor
        tokens_batch = torch.stack(tokens, dim=0)
        labels_batch = torch.stack(labels, dim=0)
        guis_batch = torch.stack(guis, dim=0)
        
        return{
            'token': tokens_batch,
            'label': labels_batch,
            'gui': guis_batch,
            'txt': txt_batch
        }
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pix2code_collate_fn)
    test_loader = DataLoader(dataset = test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pix2code_collate_fn)
    return train_loader, test_loader

    
