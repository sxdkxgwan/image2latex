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
    # config = Config()
    config = Test()

    max_lengths = [20, 30]
    all_scores = None

    for i, max_length in enumerate(max_lengths):
        config.logger.info("TEST: max-length = {}".format(max_length))

        # Build model
        model = Model(config)
        model.build()

        # image 
        img = imread(self.dir_images + "/" + img_path)
        img = self.img_prepro(img)


        fd = model.get_feed_dict(img, training=False, formula=formula, dropout=1)
            ce_words_eval, n_words_eval, ids_eval = sess.run(
                    [self.ce_words, self.n_words, self.pred_test.ids], feed_dict=fd)