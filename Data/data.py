import numpy as np
import os 
from Data.vocab import Vocab
from config import PAD_TOKEN, START_TOKEN, END_TOKEN
class Data:
    '''
    Stores all data of the GUIs, Sequences & Targets
    '''
    def __init__(self):
        self.vocab = Vocab()
        self.img_set = [] # holds all the guis' images
        self.sequences_set = [] # holds all the sequences
        self.targets_set = [] # holds all the targets
        self.size = 0
        self.input_shape = None
        self.output_shape = None
        self.max_length = 0
        
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
    
    def load_txt(self, paths):
        txt = []
        for p in paths:
            with open(p, 'r') as file:
                gui = file.read()
            sen = f'{START_TOKEN} {gui} {END_TOKEN}'
            sen = ' '.join(sen.split())
            sen.replace(',', ' ,')
            sen.replace('\n', ' ')
            sen.replace('\t', ' ')
            sen.replace('{', ' { ')
            sen.replace('}', ' } ')
            self.max_length = max(self.max_length, len(sen.split()))
            txt.append(sen)
            for word in gui.split():
                self.vocab.add_word(word)
        return txt
        
    def load_gui(self, paths):
        gui = []
        for p in paths:
            img = np.load(p)['features']
            gui.append(img)
        gui = np.array(gui, dtype=float)
        return gui
    
    def load_data(self, path):
        gp, n = self.load_paths(path)
        x = self.load_txt(gp)
        print('Text Loaded')
        y = self.load_gui(n)
        print('Images Loaded')
        return x, y