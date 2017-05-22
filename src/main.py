from utils.dataset import Dataset
from models.model import Model
from configs.config import config
from utils.preprocess import greyscale, get_form_prepro
from utils.data_utils import load_vocab, PAD, END


from utils.data_utils import minibatches, pad_batch_formulas, \
    pad_batch_images


if __name__ == "__main__":
    # Load vocab
    vocab = load_vocab(config.path_vocab)
    config.vocab_size = len(vocab)
    config.attn_cell_config["num_proj"] = config.vocab_size
    config.id_PAD = vocab[PAD]
    config.id_END = vocab[END]

    # Load datasets
    train_set =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_train, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(vocab))

    val_set   =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_val, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(vocab), max_iter=10)

    # Build model
    model = Model(config)
    model.build()
    model.train(val_set, val_set)