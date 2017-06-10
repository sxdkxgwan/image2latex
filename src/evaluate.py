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
    # Load config
    config = Config()

    # get dataset
    test_set  =  Dataset(path_formulas=config.path_formulas, dir_images=config.dir_images,
                    path_matching=config.path_matching_test, img_prepro=greyscale, 
                    form_prepro=get_form_prepro(config.vocab), max_len=max_length,
                    max_iter=config.max_iter)

    # Build model
    model = Model(config)
    model.build()
    model.evaluate(test_set, config.model_output, config.path_results_final, config.path_results_img)