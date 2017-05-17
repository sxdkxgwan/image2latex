import os

class Config():
    def __init__(self):
         # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

    # directories
    dir_output    = "results/test/"
    dir_images    = "../data/images_processed"
    
    path_log      = dir_output + "log.txt"

    path_matching_train = "../data/train_filter.lst"
    path_matching_val = "../data/val_filter.lst"
    path_matching_test = "../data/test_filter.lst"

    path_formulas = "../data/norm.formulas.lst"
    path_vocab    = "../data/latex_vocab.txt"
    vocab_size    = None # to be computed in main

    # preprocess images and formulas
    dim_embeddings = 100
    max_length_formula = 150
    max_shape_image = [160, 500, 1]

    # model training
    batch_size    = 10
    dropout       = 0.5
    lr            = 1e-3
    n_epochs      = 2


# initializer config
config = Config()