from utils.dataset import Dataset
from models.model import Model
from configs.config import config
from utils.preprocess import greyscale, get_form_prepro
from utils.data_utils import load_vocab



if __name__ == "__main__":
    # Load vocab
    vocab   = load_vocab(config.path_vocab)

    # Load datasets
    dataset =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(vocab))

    # Build model
    model   = Model(config)
    model.build()