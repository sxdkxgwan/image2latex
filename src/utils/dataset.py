import numpy as np
# from PIL import Image
from scipy.misc import imread
from preprocess import greyscale, get_form_prepro
import time
from data_utils import minibatches, pad_batch_images, \
    load_vocab, pad_batch_formulas


class Dataset(object):
    def __init__(self, path_formulas, dir_images, path_matching, 
                img_prepro, form_prepro, max_iter=None):
        """
        Args:
            path_formulas: (string) file of formulas, one formula per line
            dir_images: (string) dir of images, contains jpg files
            path_matchin: (string) file of name_of_img, id_formula
            img_prepro: (lambda function) takes an array -> an array
            form_prepro: (lambda function) takes a string -> array of int32
        """
        self.path_formulas = path_formulas
        self.dir_images    = dir_images
        self.path_matching = path_matching
        self.img_prepro    = img_prepro
        self.form_prepro   = form_prepro
        self.formulas      = self._load_formulas(path_formulas)
        self.length        = None
        self.max_iter      = max_iter
        

    def _load_formulas(self, filename):
        """
        Args:
            filename: (string) path of formulas, one formula per line

        Returns:
            dict: dict[idx] = one formula
        """
        formulas = dict()
        with open(filename) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                formulas[idx] = line

        return formulas


    def get_max_shape(self):
        """
        Computes max shape of images in the dataset

        Returns:
            max_shape_image: tuple (max_heigh, max_width, max_channels) 
                of images in the dataset
            max_length_formula: max length of formulas in the dataset
        """
        max_shape = [0,0,0]
        max_length = 0

        with open(self.path_matching) as f:
            for line in f:
                img_path, formula_id = line.strip().split(' ')
                img = imread(self.dir_images + "/" + img_path)
                img = self.img_prepro(img) 
                max_shape[0] = max(max_shape[0], img.shape[0])
                max_shape[1] = max(max_shape[1], img.shape[1])
                max_shape[2] = max(max_shape[2], img.shape[2])
                formula = self.form_prepro(self.formulas[int(formula_id)])
                max_length = max(max_length, len(formula))

        return max_shape, max_length


    def __iter__(self):
        """
        Iterator over Dataset

        Yields:
            img: array
            formula: one formula
        """
        with open(self.path_matching) as f:
            for idx, line in enumerate(f):
                if self.max_iter is not None and idx > self.max_iter:
                    break

                img_path, formula_id = line.strip().split(' ')
                img = imread(self.dir_images + "/" + img_path)
                img = self.img_prepro(img) 
                formula = self.form_prepro(self.formulas[int(formula_id)])
                yield img, formula


    def __len__(self):
        if self.length is None:
            counter = 0
            for _ in self:
                counter += 1
            self.length = counter

        return self.length


if __name__ == "__main__":
    path_matching = "../../data/train_filter.lst"
    dir_images    = "../../data/images_processed"
    path_formulas = "../../data/norm.formulas.lst"
    path_vocab    = "../../data/latex_vocab.txt"
    batch_size    = 10
    vocab         = load_vocab(path_vocab)
    myset = Dataset(path_formulas=path_formulas, dir_images=dir_images,
                    path_matching=path_matching, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(vocab))

    for x_batch, y_batch in minibatches(myset, batch_size):
        x_batch = pad_batch_images(x_batch, myset.max_shape_image)
        y_batch, y_length = pad_batch_formulas(y_batch, myset.max_length_formula)
        print x_batch.shape, y_batch.shape
