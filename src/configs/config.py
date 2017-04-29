class Config():
    # directories
    dir_output    = "results/test/"
    dir_images    = "../data/images_processed"
    
    path_log      = dir_output + "log.txt"
    path_matching = "../data/val_filter.lst"
    path_formulas = "../data/norm.formulas.lst"
    path_vocab    = "../data/latex_vocab.txt"

    # model training
    batch_size    = 10

config = Config()