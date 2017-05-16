import numpy as np
from data_utils import UNK


def greyscale(state):
    """
    Preprocess state (:, :, 3) image into
    """
    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)


def get_form_prepro(vocab):
    """
    Args:
        vocab: dict[token] = id
    Returns:
        lambda function(formula) -> list of ids
    """
    def get_token_id(token):
        if token in vocab:
            return vocab[token]
        else:
            return vocab[UNK]

    def f(formula):
        """
        Args:
            formula: (string)
        """
        # tokenize
        formula = formula.strip().split(' ')
        return map(lambda t: get_token_id(t), formula)


    return f
