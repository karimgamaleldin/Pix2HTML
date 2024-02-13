from Vocab import Vocab, START_TOKEN, END_TOKEN, PAD_TOKEN
import numpy as np
import os
from config import CONTEXT_SIZE

# Dataset class
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
  

training_path = './dataset/training_set'
data = Data()
train_txt, train_gui = data.load_data(training_path)
print(len(train_txt), len(train_gui))
print(train_txt[0])
print(train_gui[0].shape)

data.vocab.save_vocab('./')
# np.savez_compressed(f"./dataset/train_img.npz", features=train_gui)

outer_sen = []
outer_targets = []
max = 0
for sen in train_txt:
  prefix = [PAD_TOKEN] * CONTEXT_SIZE
  prefix.extend(sen.split())
  prefix = [data.vocab.word2index[word] for word in prefix]
  inner_sen = []
  inner_targets = []
  if len(prefix) > max:
    max = len(prefix)
  for i in range(len(prefix) - CONTEXT_SIZE):
    context = prefix[i:i+CONTEXT_SIZE]
    target = prefix[i+CONTEXT_SIZE]
    inner_sen.append(context)
    inner_targets.append(target)
  outer_sen.append(np.array(inner_sen))
  outer_targets.append(inner_targets)
print('-----------------------------------')
print(len(outer_sen), len(outer_targets))
print(outer_sen[0][1])
print(outer_targets[0])
print(max)

print(np.array(outer_targets).shape)

