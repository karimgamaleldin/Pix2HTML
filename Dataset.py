from Vocab import Vocab, START_TOKEN, END_TOKEN, PAD_TOKEN
import numpy as np
import os
from config import CONTEXT_SIZE

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
      target = prefix[i+CONTEXT_SIZE]
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
    
    # Setting input and output shape
    self.input_shape = self.image_set[0].shape
    self.output_size = len(self.vocab.word2index)

    print('Dataset loaded')

  
  def one_hot_encode(self, labels=True):
    encoded = []
    if labels: # Get the one hot encoding for each label
      for label in self.target_set:
        encoded.append(self.vocab.one_hot_encodings[label])
    else:
      for sequence in self.sequence_set: # Get the one hot encoding for each word in the sequence
        sequence_encoded = []
        for word in sequence:
          sequence_encoded.append(np.array(self.vocab.one_hot_encodings[word]))
        encoded.append(np.array(sequence_encoded))
    return np.array(encoded)
  
  def getArrays(self):
    return np.array(self.image_set), self.one_hot_encode(labels=False), self.one_hot_encode(labels=True)

  
# dataset = Dataset()
# # dataset.load_paths('./dataset/training_set')
# # print(len(set(dataset.load_paths('./dataset/test_set')[1])))
# # print(len(set(dataset.load_paths('./dataset/test_set')[0])))

# dataset.add_example('./dataset/training_set/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.gui', dataset.load_image('./dataset/training_set/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.npz'))

# print(dataset.vocab.word2index)
# print('------------------------------------------------')
# print(dataset.vocab.index2word)
# print('------------------------------------------------')
# print('------------------------------------------------')
# print(dataset.vocab.size)
# print('------------------------------------------------')
# print(dataset.sequence_set)
# print('------------------------------------------------')
# print(dataset.target_set)
# print('------------------------------------------------')
# print(dataset.size)
# print('------------------------------------------------')
# print('------------------------------------------------')
# flag = True
# print(dataset.image_set[0].shape)
# for ele in dataset.image_set:
#   flag = flag and (np.equal(ele, dataset.image_set[0])).all()
# print(flag)

# print('------------------------------------------------')
# x, y, z = dataset.getArrays()
# print(x.shape, len(dataset.sequence_set))
# print('------------------------------------------------')
# print(y.shape, len(dataset.vocab.word2index))
# print('------------------------------------------------')
# print(z.shape)
# print('------------------------------------------------')
  

dataset = Dataset()
dataset.load_dataset('./dataset/training_set')
print(dataset.size)
print('------------------------------------------------')
print(dataset.input_shape)
print('------------------------------------------------')
print(dataset.output_size)
print('------------------------------------------------')
print(dataset.vocab.size)
print('------------------------------------------------')
print(len(dataset.sequence_set))
