from vocab import Vocab, START_TOKEN, END_TOKEN, PAD_TOKEN
import numpy as np
import os
from config import CONTEXT_SIZE, IMAGE_SIZE, BATCH_SIZE
import shutil
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

# Dataset class
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

# Create data loader function
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



TRAINING_SET_NAME = "training_set"
TEST_SET_NAME = "test_set"
output_path = './dataset'
input_path = './dataset/all_data'

# Data Class that has all information about the data
class Data:
  def __init__(self):
    self.vocab = Vocab()
    self.image_set = [] # holds all the images that we need to train on
    self.sequence_set = [] # holds all the sequences that we need to train on
    self.target_set = [] # holds the letters we need to predict
    self.size = 0
    self.input_shape = None
    self.output_size = None

  def load_paths(self, path):
    print('Loading paths ....')
    gui_paths = []
    image_paths = []
    for f in os.listdir(path):
      if f.endswith(".gui"):
        path_gui = f'{path}/{f}'
        path_img = f'{path}/{f[:-4]}.npz'
        if os.path.exists(path_img):
          gui_paths.append(path_gui)
          image_paths.append(path_img)
    print('Paths loaded')
    return gui_paths, image_paths

  def load_text(self, paths):
    text = []
    for p in paths:
      with open(p, 'r') as file:
        gui = file.read()
      sen = f'{START_TOKEN} ' + gui + f' {END_TOKEN}'
      sen = ' '.join(sen.split())
      sen.replace(',', ' ,')
      sen.replace('\n', ' ')
      sen.replace('\t', ' ')
      sen.replace('{', ' { ')
      sen.replace('}', ' } ')
      text.append(sen)
      for word in gui.split():
        self.vocab.add_word(word)
    return text

  def load_gui(self, paths):
    gui = []
    for p in paths:
      img = np.load(p)['features']
      gui.append(img)
    gui = np.array(gui, dtype=float)
    return gui

  def load_data(self, path):
    gp, n = self.load_paths(path)
    x = self.load_text(gp)
    print('Text loaded')
    y = self.load_gui(n)
    return x, y
  

# Create the data directories and preprocess the images from .gui --> .npz
def create_preprocess_distribute_data():
  # Create the training set folder
  '''
  Distributes the dataset into 2 files training_set and test_set
  '''
  if not os.path.exists(f"{output_path}/{TRAINING_SET_NAME}"):
    os.makedirs(f"{output_path}/{TRAINING_SET_NAME}")
  
  # Create the test set folder
  if not os.path.exists(f"{output_path}/{TEST_SET_NAME}"):
    os.makedirs(f"{output_path}/{TEST_SET_NAME}")

  # Get the length of the dataset
  dataset_length = len(os.listdir(output_path)) / 2
  training_set_length = 1500.0
  test_set_length = dataset_length - training_set_length
  print(f"Dataset length: {dataset_length}")  
  print(f"Training set length: {training_set_length}")
  print(f"Test set length: {test_set_length}")

  paths = [] 
  for f in os.listdir(input_path):
    if f.endswith(".png"):
      paths.append(f[:-4])
  print('Number of paths', len(paths))

  # Create the training set and eval set
  i = 0
  for path in paths:
    image_path = f"{input_path}/{path}.png"
    gui_path = f"{input_path}/{path}.gui"
    img = Image.open(image_path)
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    img_rgb = img_resized.convert('RGB')
    img_array = np.array(img_rgb, dtype="float32")
    img_array = img_array / 255.0
    img = img_array
    if i < training_set_length:
      np.savez_compressed(f"{output_path}/{TRAINING_SET_NAME}/{path}.npz", features=img)
      shutil.copyfile(gui_path, f"{output_path}/{TRAINING_SET_NAME}/{path}.gui")
      retrieved_img = np.load(f"{output_path}/{TRAINING_SET_NAME}/{path}.npz")['features']
      assert np.array_equal(retrieved_img, img)
    else:
      np.savez_compressed(f"{output_path}/{TEST_SET_NAME}/{path}.npz", features=img)
      shutil.copyfile(gui_path, f"{output_path}/{TEST_SET_NAME}/{path}.gui")
      retrieved_img = np.load(f"{output_path}/{TEST_SET_NAME}/{path}.npz")['features']
      assert np.array_equal(retrieved_img, img)
    i += 1
    print(f"Processed {i} Training examples")   
