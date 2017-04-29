# Img2Latex


## Data and Preprocessing

We use Harvard preprocessing scripts that can be found at http://lstm.seas.harvard.edu/latex/

First, crop + downsampling of images + group by similar shape


```
python scripts/preprocessing/preprocess_images.py --input-dir data/sample/images --output-dir data/sample/images_processed
```

Second, parse formulas with KaTeX parser

```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/im2latex_formulas.lst --output-file data/norm.fomulas.lst
```