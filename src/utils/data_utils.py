import numpy as np
from PIL import Image
import time

UNK = "_UNK"
PAD = "_PAD"


def render(arr):
    """
    Render an array as an image
    Args:
        arr: np array (np.uint8) representing an image
    """
    img = Image.fromarray(arr)
    img.show()


def get_max_shape(arrays):
    """
    Args:
        images: list of arrays
    """
    shapes = map(lambda x: list(x.shape), arrays)
    ndim = len(arrays[0].shape)
    max_shape = []
    for d in range(ndim):
        max_shape += [max(shapes, key=lambda x: x[d])[d]]

    return max_shape


def pad_batch_images(images, target_shape):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad
    """

    # 1. max shape
    # max_shape = get_max_shape(images)
    max_shape = target_shape

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)


def pad_batch_formulas(formulas, max_length):
    """
    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas
    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32
    """
    # max_len = max(map(lambda x: len(x), formulas))
    max_len = max_length
    batch_formulas = np.zeros([len(formulas), max_len], dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula, dtype=np.int32)
        formula_length[idx] = len(formula)
        
    return batch_formulas, formula_length


def minibatches(data_generator, minibatch_size):
    """
    Args:
        data_generator: generator of (img, formulas) tuples
        minibatch_size: (int)
    Returns: 
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data_generator:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def load_vocab(filename):
    """
    Args:
        filename: (string) path to vocab txt file one word per line
    Returns:
        dict: d[token] = id
    """
    vocab = dict()
    with open(filename) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            vocab[token] = idx

    # add pad and unk tokens
    vocab[PAD] = len(vocab)
    vocab[UNK] = len(vocab)

    return vocab