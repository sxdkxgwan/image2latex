# Img2Latex


## Data and Preprocessing

We use Harvard preprocessing scripts that can be found at http://lstm.seas.harvard.edu/latex/

First, crop + downsampling of images + group by similar shape


```
python scripts/preprocessing/preprocess_images.py --input-dir data/formula_images --output-dir data/images_processed
```

Second, parse formulas with KaTeX parser

```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/im2latex_formulas.lst --output-file data/norm.formulas.lst
```

Third, filter formulas

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images_processed --label-path data/norm.formulas.lst --data-path data/im2latex_train.lst  --output-path data/train_filter.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images_processed --label-path data/norm.formulas.lst --data-path data/im2latex_validate.lst  --output-path data/val_filter.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images_processed --label-path data/norm.formulas.lst --data-path data/im2latex_test.lst  --output-path data/test_filter.lst
```


## Train

Edit the config file in configs/

```
python main.py
```