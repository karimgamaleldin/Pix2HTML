from config import *
import torch
from utils import arr2index, one_hot_encode, pad_txt
from data.Vocab import START_TOKEN, END_TOKEN
import torch.nn.functional as F

def sample_greedy(model, img, word2index, context=CONTEXT_SIZE, device='cpu', data_obj=None):
    model.to(device)
    img = img.unsqueeze(0) # put batch dimension
    img = img.to(device)
    
    # Generating sequence
    seq = [START_TOKEN]
    out = f'{START_TOKEN} '
    
    for i in range(180): ## as the largest in the dataset is 90 so x2.
        if len(seq) > context:
            seq = seq[-context:]
        
        ind_seq = arr2index(seq)
        padded = torch.tensor(one_hot_encode(pad_txt(ind_seq, context), word2index)).unsqueeze(0).float()
        # print(padded.shape)
        padded = padded.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(img, padded)
            # print(logits)
            probs = F.softmax(logits, dim=-1)
            probs_max = probs.argmax(1)
        model.train()
        out_token = data_obj.vocab.index2word[probs_max.item()]
        seq.append(out_token)
        out += out_token + " "
        if out_token == END_TOKEN:
            break
            
    return out