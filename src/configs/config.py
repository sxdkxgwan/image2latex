import shutil
from utils.data_utils import load_vocab, PAD, END
import os


class Config():
    def __init__(self):
        """
        Creates output directories if they don't exist and load vocabulary
        Defines attributes that depends on the vocab.
        Look for the __init__ comments in the class attributes
        """
        # check that the reload directory exists
        if self.dir_reload is not None and not os.path.exists(self.dir_reload):
            print("Weights directory not found ({})".format(self.dir_reload))
            self.dir_reload = None

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        else:
            print("ERROR: Results directory from previous experience. Abort.")
            raise Exception

        if not os.path.exists(self.model_output):
            os.makedirs(self.model_output)

        # initializer file for answers by erasing previous files
        with open(self.path_results, "w") as f:
            pass

        with open(self.path_results_final, "w") as f:
            pass

        self.vocab = load_vocab(self.path_vocab)
        self.vocab_size = len(self.vocab)
        self.attn_cell_config["num_proj"] = self.vocab_size
        self.id_PAD = self.vocab[PAD]
        self.id_END = self.vocab[END]


    # directories
    dir_output    = "results/results_50/"
    dir_images    = "../data/images_processed"
    
    path_log            = dir_output + "log.txt"
    path_results        = dir_output + "results_val.txt"
    model_output        = dir_output + "model.weights/"
    path_results_final  = dir_output + "results.txt"
    path_results_img    = dir_output + "images/"
    # dir_reload          = "results/session_init/model.weights/"
    dir_reload          = None

    path_matching_train = "../data/train_filter.lst"
    path_matching_val = "../data/val_filter.lst"
    path_matching_test = "../data/test_filter.lst"

    path_formulas = "../data/norm.formulas.lst"

    # vocab things
    path_vocab    = "../data/latex_vocab.txt"
    path_embeddings = "../data/embeddings.npz"
    pretrained_embeddings = False
    trainable_embeddings = True
    vocab_size    = None # to be computed in __init__
    id_PAD        = None # to be computed in __init__
    id_END        = None # to be computed in __init__

    # preprocess images and formulas
    dim_embeddings = 80
    max_length_formula = 50

    # model training parameters
    n_epochs      = 15
    batch_size    = 20
    dropout       = 1 # keep_prob
    max_iter      = None

    # learning rate stuff
    lr_init       = 1e-3
    lr_min        = 1e-4
    start_decay   = 10 # start decaying from 11th epoch
    end_decay     = 14 # end decay at 15th decay and stay at lr_min
    decay_rate    = 0.5 # decay rate if perf does not improve
    lr_warm       = 1e-4 # warm up with lower learning rate because of high gradients
    end_warm      = 2 # keep warm up for 2 epochs

    # encoder
    encoder_dim = 256
    encode_with_lstm = False

    # decoder
    attn_cell_config = {
        "cell_type": "lstm",
        "num_units": 512,
        "dim_e": 512,
        "dim_o": 512,
        "num_proj": None, # to be computed in __init__  because vocab size
        "dim_embeddings": dim_embeddings
    }
    decoding = "beam_search" # or "beam_search"
    beam_size = 5


class Test(Config):
    n_epochs = 200
    batch_size = 20
    max_iter = 40
    max_length_formula = 20
    decoding = "beam_search"
    encode_with_lstm = False