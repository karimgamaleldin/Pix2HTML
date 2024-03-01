import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from Data.data import Data
from Data.vocab import Vocab
from config import PAD_TOKEN, START_TOKEN, END_TOKEN, CONTEXT_SIZE

class Pix2HTMLDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        
        # Getting the data
        self.data_obj = Data()
        self.train_txt, self.train_gui = self.data_obj.load_data(self.train_dir)
        self.test_txt, self.test_gui = self.data_obj.load_data(self.test_dir)
        self.vocab_size = self.data_obj.vocab.size
        
        # One hot enccoding our data
        train_one_hot = self.one_hot_all_data(self.train_txt, self.data_obj.vocab)
        test_one_hot = self.one_hot_all_data(self.test_txt, self.data_obj.vocab)
        
        # Padding and making our sequences and labels
        train_seq, train_labels = self.create_labels_seq(train_one_hot, self.data_obj.max_length, self.data_obj.vocab)
        test_seq, test_labels = self.create_labels_seq(test_one_hot, self.data_obj.max_length, self.data_obj.vocab)
        train_seq = torch.tensor(train_seq).float()
        train_labels = torch.tensor(train_labels).float()
        test_seq = torch.tensor(test_seq).float()
        test_labels = torch.tensor(test_labels).float()
        
        #  Permuting/Transposing our images to make the channels first
        train_gui = torch.tensor(self.train_gui.transpose((0, 3, 1, 2))).float()
        test_gui = torch.tensor(self.test_gui.transpose((0, 3, 1, 2))).float()
        
        # Creating our dataset
        self.train_dataset = Pix2HTMLDataset(train_seq, train_labels, train_gui, self.train_txt)
        self.test_dataset = Pix2HTMLDataset(test_seq, test_labels, test_gui, self.test_txt)
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate)
    
    def one_hot_all_data(self, arr, vocab: Vocab):
        vocab.create_one_hot_encoding()
        res = []
        for sen in arr:
            res.append([vocab.one_hot_encodings[word] for word in sen.split()])
        return res
    
    def pad_txt(self, arr, max_length, vocab : Vocab):
        pad_one_hot = vocab.one_hot_encodings[PAD_TOKEN]
        pad_seq = [pad_one_hot] * max_length
        pad_seq[-len(arr):] = arr
        return pad_seq
    
    def create_labels_seq(self, data, max_length: int, vocab: Vocab):
        seq = []
        targets = []
        for sen in data:
            padded = pad_txt(sen, max_length, vocab)
            for i in range(max_length - CONTEXT_SIZE):
                context = padded[i:i+CONTEXT_SIZE]
                target = padded[i+CONTEXT_SIZE]
                seq.append(context)
                targets.append(target)
        return seq, targets
    
    def custom_collate(self, batch):
        # Getting all the items in an array
        seqs = [item['seq'] for item in batch]
        labels = [item['label'] for item in batch]
        guis = [item['gui'] for item in batch]
        txt_batch = [item['txt'] for item in batch]
        
        # stacking the tensor
        seqs_batch = torch.stack(seqs, dim=0)
        labels_batch = torch.stack(seqs, dim=0)
        guis_batch = torch.stack(seqs, dim=0)

        return {
        'seq': seqs_batch,
        'label': labels_batch,
        'gui': guis_batch,
        'txt': txt_batch
    }
        
    

class Pix2HTMLDataset(Dataset):
    def __init__(self, seqs, labels, guis, txts):
        self.seqs = seqs
        self.labels = labels
        self.guis = guis
        self.txts = txts
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        seq = self.seqs[index]
        label = self.labels[index]
        gui = self.guis[index // 42]
        txt = self.txts[index // 42]
        
        return {
            'seq': seq,
            'label': label,
            'gui': gui,
            'txt': txt
        }