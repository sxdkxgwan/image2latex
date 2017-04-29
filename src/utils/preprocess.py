import numpy as np


def greyscale(state):
    """
    Preprocess state (:, :, 3) image into
    """
    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    return state.astype(np.uint8)


def get_form_prepro(vocab):
    """
    Args:
        vocab: dict[token] = id
    Returns:
        lambda function(formula) -> list of ids
    """

    def f(formula):
        """
        Args:
            formula: (string)
        """
        # tokenize
        formula = formula.strip().split(' ')
        return map(lambda t: vocab[t], formula)


    return f
