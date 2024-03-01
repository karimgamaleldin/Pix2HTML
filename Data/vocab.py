import numpy as np
from config import PAD_TOKEN, START_TOKEN, END_TOKEN

# Vocab object
class Vocab:
    '''
    Stores all the data related to our vocab
    '''
    def __init__(self):
        self.index2word = {}
        self.word2index = {}
        self.one_hot_encodings = {}
        self.size = 0
        self.add_word(PAD_TOKEN)
        self.add_word(START_TOKEN)
        self.add_word(END_TOKEN)
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.index2word[self.size] = word
            self.size += 1
            
            
    def save_vocab(self, path):
        output_file = f'{path}/words.vocab'
        with open(output_file, 'w') as file:
            for word in self.word2index:
                file.write(f"{word} {self.word2index[word]}\n")
                
    def load_vocab(self, path):
        input_file = f'{path}/words.vocab'
        with open(input_file, 'r') as file:
            for line in file:
                word, index = line.split()
                self.word2index[word] = int(index)
                self.index2word[int(index)] = word
                self.size += 1 
            
    def create_one_hot_encoding(self):
        for key, value in self.word2index.items():
            one_hot = np.zeros(self.size)
            one_hot[value] = 1
            self.one_hot_encodings[key] = one_hot