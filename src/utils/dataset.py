import numpy as np
from PIL import Image
from preprocess import greyscale, get_form_prepro
import time
from data_utils import minibatches, pad_batch_images, \
    load_vocab, pad_batch_formulas


class Dataset(object):
    def __init__(self, path_formulas, dir_images, path_matching, 
                img_prepro, form_prepro):
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


    def __iter__(self):
        """
        Iterator over Dataset

        Yields:
            img: array
            formula: one formula
        """
        with open(self.path_matching) as f:
            for line in f:
                img_path, formula_id = line.strip().split(' ')
                img = Image.open(self.dir_images + "/" + img_path)
                img = np.asarray(img)
                img = self.img_prepro(img)
                formula = self.form_prepro(self.formulas[int(formula_id)])
                yield img, formula




if __name__ == "__main__":
    path_matching = "../../data/val_filter.lst"
    dir_images    = "../../data/images_processed"
    path_formulas = "../../data/norm.formulas.lst"
    path_vocab    = "../../data/latex_vocab.txt"
    batch_size    = 10
    vocab         = load_vocab(path_vocab)
    myset = Dataset(path_formulas=path_formulas, dir_images=dir_images,
                    path_matching=path_matching, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(vocab))

    for x_batch, y_batch in minibatches(myset, batch_size):
        print len(x_batch)
        print len(y_batch)
        for x in x_batch:
            print x.shape

        for y in y_batch:
            print len(y)

        x_batch = pad_batch_images(x_batch)
        y_batch = pad_batch_formulas(y_batch)
        print y_batch
        print x_batch.shape
        print y_batch.shape
        
        time.sleep(5)
