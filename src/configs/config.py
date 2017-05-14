import os

class Config():
    # directories
    dir_output    = "results/test/"
    dir_images    = "../data/images_processed"
    
    path_log      = dir_output + "log.txt"

    path_matching_train = "../data/train_filter.lst"
    path_matching_val = "../data/val_filter.lst"
    path_matching_test = "../data/test_filter.lst"

    path_formulas = "../data/norm.formulas.lst"
    path_vocab    = "../data/latex_vocab.txt"
    vocab_size    = None

    #preprocess images and formulas
    max_length_formula = 150
    max_shape_image = [160,520,1]

    # model training
    batch_size    = 10
    dropout       = 0.5
    lr            = 1e-3
    n_epochs      = 2

    #encoder parameters
    encoder_hidden_size = 256 
    
    #decoder parameters
    decoder_hidden_size = 512

    def __init__(self):
         # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

config = Config()