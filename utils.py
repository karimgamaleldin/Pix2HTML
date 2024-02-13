from config import CONTEXT_SIZE
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(train_loss, test_loss):
    train_loss = [loss.cpu().item() for loss in train_loss]
    test_loss = [loss.cpu().item() for loss in test_loss]
    x = np.arange(0, len(train_loss))
    plt.plot(x, train_loss, c='b', label='Train Loss')
    plt.plot(x, test_loss, c='y', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('loss curves')
    plt.show()

    
def perm_channels(images):
    out = []
    for i, img in enumerate(images):
        img = img.transpose((2, 0, 1))
        out.append(img)
    return out

def indexify(d):
    max_count = 0
    new = []
    for i in range(len(d)):
        new.append([data_obj.vocab.word2index[word] for word in d[i].split()])
        if len(new[i]) > max_count:
            max_count = len(new[i])
    return new, max_count

def make_tokens(data, max_count, word2index):
    outer_sen = []
    outer_labels = []

    for l in data:
        inner_sen = []
        inner_labels = []
        padded = pad_txt(l, max_count)
        for i in range(len(padded) - CONTEXT_SIZE):
            context = padded[i:i+CONTEXT_SIZE]
            target = padded[i+CONTEXT_SIZE]
            inner_sen.append(one_hot_encode(context, word2index))
            inner_labels.append(one_hot_encode([target], word2index)[0])
        outer_sen.append(inner_sen)
        outer_labels.append(inner_labels)
    return outer_sen, outer_labels
        
def one_hot_encode(data, word2index):
    res = []
    for ele in data:
        enc = [0] * len(word2index)
        enc[int(ele)] = 1
        res.append(enc)
    return res
def pad_txt(idxs, max_count):
    pad_seq = np.zeros(max_count, dtype=np.float32)
    pad_seq[-len(idxs):] = idxs
    return pad_seq

def arr2index(arr, data_obj):
    res = []
    for word in arr:
        res.append(data_obj.vocab.word2index[word])
    return res