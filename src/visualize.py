from utils.dataset import Dataset
from models.model import Model
from configs.config import Config, Test
from utils.preprocess import greyscale, get_form_prepro, compose
from utils.data_utils import minibatches, pad_batch_formulas, \
    pad_batch_images
from utils.lr_schedule import LRSchedule
import tensorflow as tf
from utils.evaluate import simple_plots


if __name__ == "__main__":
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    config = Config() # for test purposes


    test_set  =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_test, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(config.vocab), max_len=config.max_length_formula,
                    max_iter=config.max_iter)


    lengths = []
    for _, formula in test_set:
        lengths.append(len(formula))

    plt.figure()
    plt.hist(lengths, 10, histtype='bar', rwidth=0.8)
    plt.xlabel("Formula Length")
    plt.ylabel("Counts")
    plt.savefig("form_length.png")
    plt.close()

