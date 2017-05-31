from collections import Counter
import numpy as np
import nltk
from utils.data_utils import reconstruct_formula, END
from subprocess import call
import os
import PIL
from PIL import Image
import distance


def evaluate(references, hypotheses, rev_vocab, path, id_END):
    """
    Evaluate BLEU and EM scores from txt hypotheses and references
    Write answers in a text file

    Args:
        references: list of lists of list (multiple references per hypothesis)
        hypotheses: list of list
        rev_vocab: (dict) rev_vocab[idx] = word
        path: (string) path where to write results
    """
    hypotheses = truncate_end(hypotheses, id_END)
    write_answers(references, hypotheses, rev_vocab, path)
    scores = dict()
    scores["BLEU-4"] = bleu_score(references, hypotheses)
    scores["EM"] = exact_match_score(references, hypotheses)
    return scores
    

def truncate_end(hypotheses, id_END):
    """
    Dummy code to remove the end of each sentence starting from
    the first id_END token.
    """
    trunc_hypotheses = []
    for hypo in hypotheses:
        trunc_hypo = []
        for id_ in hypo:
            if id_ == id_END:
                break
            trunc_hypo.append(id_)
        trunc_hypotheses.append(trunc_hypo)

    return trunc_hypotheses



def write_answers(references, hypotheses, rev_vocab, path):
    """ 
    Write text answers in file, the format is
        truth
        prediction
        new line
        ...
    """
    assert len(references) == len(hypotheses)

    with open(path, "a") as f:
        for refs, hypo in zip(references, hypotheses):
            ref = refs[0] # only take first ref
            ref = [rev_vocab[idx] for idx in ref]
            hypo = [rev_vocab[idx] for idx in hypo]
            f.write(" ".join(ref) + "\n")
            f.write(" ".join(hypo) + "\n\n")


def exact_match_score(references, hypotheses):
    """
    Compute exact match scores.

    Args:
        references: list of list of list of ids of tokens
            (assumes multiple references per exemple). In
            our case we only consider the first reference.

        hypotheses: list of list of ids of tokens
    """
    exact_match = 0
    for refs, hypo in zip(references, hypotheses):
        ref = refs[0] # only take first ref
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """
    Computes bleu score. BLEU-4 has been shown to be the most 
    correlated with human judgement so we use this one.
    """
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def img_edit_distance(file1, file2):
    """
    Computes Levenshtein distance between two images.
    Slice the images into columns and consider one column as a character.

    Code strongly inspired by Harvard's evaluation scripts.

    Args:
        file1: (string) path to image (reference)
        file2: (string) path to image (hypothesis)
    Returns:
        column wise levenshtein distance
    """
    # load the image
    im1 = Image.open(file1).convert('L')
    im2 = Image.open(file2).convert('L')

    # transpose and convert to 0 or 1
    img_data1 = np.asarray(im1, dtype=np.uint8) # height, width
    img_data1 = np.transpose(img_data1)
    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]
    img_data1 = (img_data1<=128).astype(np.uint8)

    img_data2 = np.asarray(im2, dtype=np.uint8) # height, width
    img_data2 = np.transpose(img_data2)
    h2 = img_data2.shape[1]
    w2 = img_data2.shape[0]
    img_data2 = (img_data2<=128).astype(np.uint8)

    # create binaries for each column
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    elif h1 > h2:# pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for item in img_data2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]

    # convert each column binary into int
    seq1_int = [int(item,2) for item in seq1]
    seq2_int = [int(item,2) for item in seq2]

    # compute distance
    edit_distance = distance.levenshtein(seq1_int, seq2_int)

    return edit_distance


def pad_image(img, output_path, pad_size=[8,8,8,8]):
    """
    Pads image with pad size

    Args:
        img: (string) path to image
        output_path: (string) path to output image
    """
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0]+PAD_LEFT+PAD_RIGHT, old_im.size[1]+PAD_TOP+PAD_BOTTOM)
    new_size = old_size
    new_im = Image.new("RGB", new_size, (255,255,255))
    new_im.paste(old_im, (PAD_LEFT,PAD_TOP))
    new_im.save(output_path)


def crop_image(img, output_path):
    """
    Crops image to content

    Args:
        img: (string) path to image
        output_path: (string) path to output image
    """
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        old_im.save(output_path)
        return False

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    old_im.save(output_path)
    return True


def downsample_image(img, output_path, ratio=2):
    """
    Downsample image by ratio
    """
    assert ratio>=1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True


def convert_to_png(formula, path_out, name):
    """
    Convert latex to png image

    Args:
        formula: (string) of latex
        path_out: (string) path to output directory
        name: (string) name of file
    """
    # write formula into a .tex file
    with open(path_out + "{}.tex".format(name), "w") as f:
        f.write(
    r"""\documentclass[preview]{standalone}
    \begin{document}
        $$ %s $$
    \end{document}""" % (formula))

    # call pdflatex to create pdf
    call(["pdflatex", "-interaction=nonstopmode", "-output-directory", path_out, path_out+"{}.tex".format(name)])

    # call magick to convert the pdf into a png file
    call(["magick",  "convert",  "-density", "200", "-quality", "100", path_out+"{}.pdf".format(name), path_out+"{}.png".format(name)])

    # cleaning
    os.remove(path_out+"{}.aux".format(name))
    os.remove(path_out+"{}.log".format(name))
    os.remove(path_out+"{}.pdf".format(name))
    os.remove(path_out+"{}.tex".format(name))

    # crop, pad and downsample
    img_path = path_out + "{}.png".format(name)
    crop_image(img_path, img_path)
    pad_image(img_path, img_path)
    downsample_image(img_path, img_path)


def evaluate_images_and_edit(path_in, path_out):
    """
    Render latex formulas into png of reference and hypothesis

    Args:
        path: path of results.txt
    Returns:
        levenhstein distance between formulas
        levenhstein distance between columns of rendered images
    """
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    with open(path_in) as f:
        em_txt = em_img = total_txt = edit_txt = total_img = edit_img = 0
        ref, hypo = None, None
        for i, line in enumerate(f):
            if line == "\n":
                ref, hypo = None, None
            if ref is None and hypo is None:
                ref = line.strip()
                continue
            if ref is not None and hypo is None:
                hypo = line.strip()
                print("Generating formula {}".format(i/2 + 1))

                try:
                    convert_to_png(ref, path_out, "{}_ref".format(i/2 + 1))
                    convert_to_png(hypo, path_out, "{}_hypo".format(i/2 + 1))
                    
                    tokens_ref, tokens_hypo = ref.split(' '), hypo.split(' ')

                    edit_txt  += distance.levenshtein(tokens_ref, tokens_hypo)
                    edit_img  += img_edit_distance(path_out+"{}_ref.png".format(i/2 + 1), path_out+"{}_hypo.png".format(i/2 + 1))
                    
                    # exact matches = when edit distance == 0
                    if edit_img == 0:
                        em_img += 1
                    if edit_txt == 0:
                        em_txt += 1

                    # increment total counts
                    total_txt += 1
                    total_img += 1

                except Exception, e:
                    print("ERROR")
                    print(e)

                    try:
                        tokens_ref, tokens_hypo = ref.split(' '), hypo.split(' ')
                        
                        edit_txt += distance.levenshtein(tokens_ref, tokens_hypo)
                        # exact matches
                        if edit_txt == 0:
                            em_txt += 1
                        # increment total count
                        total_txt += 1

                    except Exception, e:
                        print(e)


        scores = dict()
        scores["Levenshtein Text"] = edit_txt / float(max(total_txt, 1))
        scores["Levenshtein Img"]  = edit_img / float(max(total_img, 1))
        scores["EM Text"]          = em_txt / float(max(total_txt, 1))
        scores["EM Img"]           = em_img / float(max(total_img, 1))

        info = "Unable to render LaTeX for {} out of {} images".format(total_txt - total_img, total_txt)

        return scores, info
