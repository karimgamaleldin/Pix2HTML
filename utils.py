from config import CONTEXT_SIZE
import numpy as np

# utils
def perm_channels(images):
    out = []
    for i, img in enumerate(images):
        img = img.transpose((2, 0, 1))
        out.append(img)
    return out

def indexify(d, data):
    max_count = 0
    new = []
    for i in range(len(d)):
        new.append([data.vocab.word2index[word] for word in d[i].split()])
        if len(new[i]) > max_count:
            max_count = len(new[i])
    return new, max_count

def make_tokens(data):
    outer_sen = []
    outer_labels = []

    for l in data:
        inner_sen = []
        inner_labels = []
        padded = pad_txt(l)
        for i in range(len(padded) - CONTEXT_SIZE):
            context = padded[i:i+CONTEXT_SIZE]
            target = padded[i+CONTEXT_SIZE]
            inner_sen.append(context)
            inner_labels.append(target)
        outer_sen.append(inner_sen)
        outer_labels.append(inner_labels)
    return outer_sen, outer_labels
        
def pad_txt(idxs, max_count):
    pad_seq = np.zeros(max_count, dtype=np.float32)
    pad_seq[-len(idxs):] = idxs
    return pad_seq