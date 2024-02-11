from Vocab import Vocab, START_TOKEN, END_TOKEN, PAD_TOKEN
import numpy as np
import os

CONTEXT_SIZE = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64

class Dataset:
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
  
  def load_gui(self, path):
    with open(path, 'r') as file:
      gui = file.read()
    return gui
  
  def load_image(self, path):
    img = np.load(path)['features']
    return img
  
  def add_example(self, gui, image):
    # Tokens in gui file
    tokens = [START_TOKEN]
    # Opening gui file to read the tokens 
    with open(gui, 'r') as file:
      for line in file:
        gui = line.split()
        for word in gui:
          self.vocab.add_word(word)
          tokens.append(word)
    tokens.append(END_TOKEN)
    # Add the context and target to the dataset
    prefix = [PAD_TOKEN] * CONTEXT_SIZE
    prefix.extend(tokens)
    for i in range(len(prefix) - CONTEXT_SIZE):
      context = prefix[i:i+CONTEXT_SIZE]
      target = tokens[i+CONTEXT_SIZE]
      self.image_set.append(image)
      self.sequence_set.append(context)
      self.target_set.append(target)

  def load_dataset(self, path):
    print('Loading dataset ....')
    # Putting data into arrays
    gui_paths, image_paths = self.load_paths(path)
    self.size = len(gui_paths)
    for i in range(self.size):
      gui_path = gui_paths[i]
      image_path = image_paths[i]
      image = self.load_image(image_path)
      self.add_example(gui_path, image)
    
    # Generating one hot encodings
    self.vocab.create_one_hot_encoding()
    self.one_hot_encode()
    
    # todo: continue this method

    print('Dataset loaded')

  
  def one_hot_encode(self):
    temp = []
    for label in self.target_set:
      temp.append(self.vocab.one_hot_encodings[label])
    self.target_set = temp

  
