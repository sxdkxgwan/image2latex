from collections import Counter
import numpy as np
import nltk
from utils.data_utils import reconstruct_formula


def write_answers(references, hypotheses, rev_vocab, path):
    """ 
    Write answers in file, the format is
        truth
        prediction
        new line
        ...
    """
    assert len(references) == len(hypotheses)

    with open(path, "a") as f:
        for refs, hypo in zip(references, hypotheses):
            ref = refs[0] # only take first ref
            ref = [rev_vocab[idx] for idx in ref]
            hypo = [rev_vocab[idx] for idx in hypo]
            f.write(" ".join(ref) + "\n")
            f.write(" ".join(hypo) + "\n\n")


def f1_score(prediction, ground_truth):
    # prediction_tokens = prediction.split()
    # ground_truth_tokens = ground_truth.split()
    prediction_tokens = prediction[:len(ground_truth)]
    ground_truth_tokens = ground_truth

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(references, hypotheses):
    exact_match = 0
    for refs, hypo in zip(references, hypotheses):
        ref = refs[0] # only take first ref
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
	BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25))
	return BLEU_4


def evaluate(references, hypotheses, rev_vocab, path):
    """
    Args:
        references: list of lists of list (multiple references per hypothesis)
        hypotheses: list of list
        rev_vocab: (dict) rev_vocab[idx] = word
        path: (string) path where to write results
    """
    write_answers(references, hypotheses, rev_vocab, path)
    scores = dict()
    scores["BLEU-4"] = bleu_score(references, hypotheses)
    scores["EM"] = exact_match_score(references, hypotheses)
    return scores
    
