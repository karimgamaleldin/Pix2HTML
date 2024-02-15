import nltk
from nltk.translate.bleu_score import sentence_bleu
from data import Data
from generate import get_sampled_txt

def evaluate_avg_bleu_score(true_txt_arr, sampled_txt_arr):
    assert len(true_txt_arr) == len(sampled_txt_arr)
    score = 0
    for i in range(len(true_txt_arr)):
        reference = [true_txt_arr[i].split()]
        candidate = sampled_txt_arr[i].split()
        score += sentence_bleu(reference, candidate)
    return score / len(true_txt_arr)

def evaluate_avg_acc(true_txt_arr, sampled_txt_arr):
    '''
    Similar to the one in the paper
    '''
    assert len(true_txt_arr) == len(sampled_txt_arr)
    total_error = 0
    for i in range(len(true_txt_arr)):
        reference = true_txt_arr[i].split()
        candidate = sampled_txt_arr[i].split()
        error = abs(len(reference) - len(candidate))
        for ref, can in zip(reference, candidate):
            if ref != can:
                error += 1
        total_error += (error / len(reference))
    return 1 - (total_error / len(true_txt_arr))

def get_results(gui, txt, model, data_obj: Data):
    # Greedy (Greedy is beam = 1)
    sampled_txt_1 = get_sampled_txt(gui, model, data_obj.vocab.word2index, beam=1)
    bleu_greedy = evaluate_avg_bleu_score(txt, sampled_txt_1)
    acc_greedy = evaluate_avg_acc(txt, sampled_txt_1)
    print(f"Greedy - BLEU : {bleu_greedy}, Accuracy : {acc_greedy}")
    sampled_txt_3 = get_sampled_txt(gui, model, data_obj.vocab.word2index, beam=3)
    bleu_greedy = evaluate_avg_bleu_score(txt, sampled_txt_3)
    acc_greedy = evaluate_avg_acc(txt, sampled_txt_3)
    print(f"Beam 3 - BLEU : {bleu_greedy}, Accuracy : {acc_greedy}")
    sampled_txt_5 = get_sampled_txt(gui, model, data_obj.vocab.word2index, beam=5)
    bleu_greedy = evaluate_avg_bleu_score(txt, sampled_txt_5)
    acc_greedy = evaluate_avg_acc(txt, sampled_txt_5)
    print(f"Beam 5 - BLEU : {bleu_greedy}, Accuracy : {acc_greedy}")