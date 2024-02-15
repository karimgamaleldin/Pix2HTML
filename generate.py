from config import *
import torch
from utils import arr2index, one_hot_encode, pad_txt
from vocab import START_TOKEN, END_TOKEN
import torch.nn.functional as F
from config import device
from data import Data

def greedy_search(model, img, word2index, data_obj: Data, context=CONTEXT_SIZE, device=device):
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
        out_token = data_obj.vocab.index2word[probs_max.item()]
        seq.append(out_token)
        out += out_token + " "
        if out_token == END_TOKEN:
            break
            
    return out

def beam_search(model, img, word2index , data_obj: Data, context=CONTEXT_SIZE, device=device, beam_width=5):
    model.to(device)
    img = img.unsqueeze(0)  # put batch dimension
    img = img.to(device)
    
    # Initialize sequences with the start token
    sequences = [([START_TOKEN], 0)]
    
    for _ in range(180):
        cand = []
        for i, seq in enumerate(sequences):
            temp = seq[0]
            if len(temp) > context:
                temp = seq[0][-context:]
            
            # A completed sequence
            if temp[-1] == END_TOKEN:
                cand.append(seq)
                continue
            # Generate 
            ind_seq = arr2index(temp)
            padded = torch.tensor(one_hot_encode(pad_txt(ind_seq, context), word2index)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                model.eval()
                logits = model(img, padded)
                probs = F.softmax(logits, dim=1)
            
            # Getting the top beam_width tokens
            top_probs, top_indices = probs.topk(beam_width, dim=-1)
            for j in range(beam_width):
                next_token = top_indices[0][j].item()
                next_token_prob = top_probs[0][j].item()
                cand_seq = seq[0] + [data_obj.vocab.index2word[next_token]]
                cand_score = seq[1] + torch.log(torch.tensor(next_token_prob, device=device))
                cand.append((cand_seq, cand_score))
            
        # Sort candidates descending by score
        cand.sort(key=lambda x: x[1], reverse=True)
        sequences = cand[:beam_width]
    
    return ' '.join(sequences[0][0]) # as it is already sorted from last 2 steps


def get_sampled_txt(guis, model, word2index, beam=1):
    samples = []
    for i, g in  enumerate(guis):
        samples.append(beam_search(model, g, word2index, beam_width=beam))
    return samples
