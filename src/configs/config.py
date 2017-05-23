from utils.data_utils import load_vocab, PAD, END
from utils.lr_schedule import LRSchedule
import os

class Config():
    def __init__(self):
        """
        Creates output directories if they don't exist
        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        if not os.path.exists(self.model_output):
            os.makedirs(self.model_output)

        # initializer file for answers by erasing previous files
        with open(self.path_answers, "w") as f:
            pass

        self.vocab = load_vocab(self.path_vocab)
        self.vocab_size = len(self.vocab)
        self.attn_cell_config["num_proj"] = self.vocab_size
        self.id_PAD = self.vocab[PAD]
        self.id_END = self.vocab[END]
        self.lr_schedule = LRSchedule(self.lr_init, self.lr_min, self.start_decay, 
                                      self.decay_rate, self.n_steps)

    # directories
    dir_output    = "results/test/"
    dir_images    = "../data/images_processed"
    
    path_log      = dir_output + "log.txt"
    path_answers  = dir_output + "results.txt"
    model_output  = dir_output + "model.weights/"

    path_matching_train = "../data/train_filter.lst"
    path_matching_val = "../data/val_filter.lst"
    path_matching_test = "../data/test_filter.lst"

    path_formulas = "../data/norm.formulas.lst"

    # vocab things
    path_vocab    = "../data/latex_vocab.txt"
    vocab_size    = None # to be computed in __init__
    id_PAD        = None # to be computed in __init__
    id_END        = None # to be computed in __init__

    # preprocess images and formulas
    dim_embeddings = 100
    max_length_formula = 150
    max_shape_image = [160, 500, 1]

    # model training parameters
    n_epochs      = 50
    batch_size    = 10
    dropout       = 0.5
    max_iter      = 2

    # learning rate stuff
    lr_init       = 1e-2
    lr_min        = 1e-3
    start_decay   = 0 # start decaying from begining
    decay_rate    = 0.5 # decay rate if eval score doesn't improve
    # n_steps       = ((max_iter + batch_size - 1) // batch_size) * n_epochs
    n_steps       = None
    
    # model config
    attn_cell_config = {
        "num_units": 200,
        "dim_e": 200,
        "dim_o": 200,
        "num_proj": None, # to be computed in __init__
        "dim_embeddings": dim_embeddings
    }