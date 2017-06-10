import os
from collections import Counter
import numpy as np
import nltk
from utils.data_utils import reconstruct_formula, END
import os
import PIL
from PIL import Image
import distance
from .general import run


TIMEOUT = 10


def evaluate(references, hypotheses, rev_vocab, path, id_END):
    """
    Evaluate BLEU and EM scores from txt hypotheses and references
    Write answers in a text file

    Args:
        references: list of lists of list (multiple references per hypothesis)
        hypotheses: list of list of list (multiple hypotheses)
        rev_vocab: (dict) rev_vocab[idx] = word
        path: (string) path where to write results
    """
    hypotheses = truncate_end(hypotheses, id_END)
    write_answers(references, hypotheses, rev_vocab, path)
    scores = dict()
    
    # extract best hypothesis to compute scores
    hypotheses = [hypos[0] for hypos in hypotheses]
    scores["BLEU-4"] = bleu_score(references, hypotheses)
    scores["EM"] = exact_match_score(references, hypotheses)
    return scores
    

def truncate_end(hypotheses, id_END):
    """
    Dummy code to remove the end of each sentence starting from
    the first id_END token.
    """
    trunc_hypotheses = []
    for hypos in hypotheses:
        trunc_hypos = []
        for hypo in hypos:
            trunc_hypo = []
            for id_ in hypo:
                if id_ == id_END:
                    break
                trunc_hypo.append(id_)
            trunc_hypos.append(trunc_hypo)

        trunc_hypotheses.append(trunc_hypos)

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
        for refs, hypos in zip(references, hypotheses):
            ref = refs[0] # only take first ref
            ref = [rev_vocab[idx] for idx in ref]
            f.write(" ".join(ref) + "\n")

            for hypo in hypos:
                hypo = [rev_vocab[idx] for idx in hypo]
                to_write = " ".join(hypo)
                if len(to_write) > 0:
                    f.write(to_write + "\n")

            f.write("\n")


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

    return edit_distance, float(max(len(seq1_int), len(seq2_int)))



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

    try:
        # call pdflatex to create pdf
        run("pdflatex -interaction=nonstopmode -output-directory {} {}".format(path_out,
            path_out+"{}.tex".format(name)), TIMEOUT)

        # call magick to convert the pdf into a png file
        run("magick convert -density 200 -quality 100 {} {}".format(path_out+"{}.pdf".format(name),
            path_out+"{}.png".format(name)), TIMEOUT)

    except Exception, e:
        print(e)

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


def evaluate_images_and_edit(path_in, path_out, path_fig, prefix=""):
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

    counts_best = {
    "d_txt": [],
    "d_img": []
    }

    counts = {
    "d_txt": [],
    "d_img": []
    }


    ids_best = []

    with open(path_in) as f:
        # to store the results: "best" is the best among multiple hypotheses
        references, hypotheses, hypotheses_best = [], [], []
        em_txt = em_img = 0
        em_txt_best = em_img_best = 0
        total_txt_best = edit_txt_best = len_txt_best = 0
        total_img_best = edit_img_best = len_img_best = 0
        total_txt = edit_txt = len_txt = 0
        total_img = edit_img = len_img = 0

        nb_errors = total_rdr = 0
        ref, hypo, hypo_score, hypo_score_best = None, None, None, None

        ref_id, hypo_id = 0, 0

        for i, line in enumerate(f):
            if line == "\n":
                # if we reached the end of an hypo, record the best hypo
                if hypo_score_best is not None:
                    # rename the file of the best hypo an append best to it
                    try:
                        os.rename(
                            path_out + "{}_hypo_{}.png".format(ref_id, hypo_score_best["id"]), 
                            path_out + "{}_hypo_{}_best.png".format(ref_id, hypo_score_best["id"]))
                    except Exception, e:
                        print e

                    ids_best.append(hypo_score_best["id"])

                    edit_txt_best += hypo_score_best["d_txt"]
                    edit_img_best += hypo_score_best["d_img"]
                    len_img_best += hypo_score_best["l_img"]
                    len_txt_best += hypo_score_best["l_txt"]

                    counts_best["d_txt"].append(1.-hypo_score_best["d_txt"]/float(hypo_score_best["l_txt"]))
                    counts_best["d_img"].append(1.-hypo_score_best["d_img"]/float(hypo_score_best["l_img"]))
                    
                    # exact matches = when edit distance == 0
                    if hypo_score_best["d_img"] == 0:
                        em_img_best += 1
                    if hypo_score_best["d_txt"] == 0:
                        em_txt_best += 1

                    # increment total counts
                    total_txt_best += 1
                    total_img_best += 1

                    hypotheses_best.append(hypo_score_best["hypo"].split(" "))

                if hypo_score is not None:
                    edit_txt += hypo_score["d_txt"]
                    edit_img += hypo_score["d_img"]
                    len_img += hypo_score["l_img"]
                    len_txt += hypo_score["l_txt"]

                    counts["d_txt"].append(1.-hypo_score["d_txt"]/float(hypo_score["l_txt"]))
                    counts["d_img"].append(1.-hypo_score["d_img"]/float(hypo_score["l_img"]))
                    
                    # exact matches = when edit distance == 0
                    if hypo_score["d_img"] == 0:
                        em_img += 1
                    if hypo_score["d_txt"] == 0:
                        em_txt += 1

                    # increment total counts
                    total_txt += 1
                    total_img += 1

                    hypotheses.append(hypo_score["hypo"].split(" "))


                hypo_id = 0
                ref, hypo, hypo_score, hypo_score_best = None, None, None, None
                continue

            if ref is None and hypo is None:
                ref = line.strip()
                references.append([ref.split(" ")])
                ref_id += 1
                continue

            if ref is not None:
                hypo = line.strip()
                hypo_id += 1
                print("Generating formula {}".format(ref_id))

                try:
                    ref_name = "{}_ref".format(ref_id)
                    hypo_name = "{}_hypo_{}".format(ref_id, hypo_id)
                    convert_to_png(ref, path_out, ref_name)
                    convert_to_png(hypo, path_out, hypo_name)
                    
                    tokens_ref, tokens_hypo = ref.split(' '), hypo.split(' ')

                    d_txt  = distance.levenshtein(tokens_ref, tokens_hypo)
                    d_img, l_img = img_edit_distance(path_out+"{}.png".format(ref_name), path_out+"{}.png".format(hypo_name))

                    if hypo_score_best is None or hypo_score_best["d_img"] > d_img:
                        hypo_score_best = {
                            "id": hypo_id,
                            "d_txt": d_txt,
                            "d_img": d_img,
                            "l_img": l_img,
                            "l_txt": max(len(tokens_ref), len(tokens_hypo)),
                            "hypo": hypo
                        }

                    if hypo_score is None and hypo_id == 1:
                        hypo_score = {
                            "d_txt": d_txt,
                            "d_img": d_img,
                            "l_img": l_img,
                            "l_txt": max(len(tokens_ref), len(tokens_hypo)),
                            "hypo": hypo
                        }
                        
                    total_rdr += 1

                except Exception, e:
                    nb_errors += 1

        plot_histograms(counts, path_fig + str(prefix) + "_edit_hist")
        plot_histograms(counts_best, path_fig + str(prefix) + "_edit_hist_best")
        plot_histogram(ids_best, path_fig + str(prefix) + "_ids")

        scores = dict()
        # scores for the first proposal
        scores["Edit Text"] = 1. - edit_txt / float(max(len_txt, 1))
        scores["Edit Img"]  = 1. - edit_img / float(max(len_img, 1))
        scores["EM Text"]   = em_txt / float(max(total_txt, 1))
        scores["EM Img"]    = em_img / float(max(total_img, 1))
        scores["BLEU"]    = bleu_score(references, hypotheses)

        # scores for the best proposals
        scores["Edit Text Best"] = 1. - edit_txt_best / float(max(len_txt_best, 1))
        scores["Edit Img Best"]  = 1. - edit_img_best / float(max(len_img_best, 1))
        scores["EM Text Best"]   = em_txt_best / float(max(total_txt_best, 1))
        scores["EM Img Best"]    = em_img_best / float(max(total_img_best, 1))
        scores["BLEU Best"]    = bleu_score(references, hypotheses_best)

        info = "Unable to render LaTeX for {} out of {} images".format(nb_errors, total_rdr)

        return scores, info


def plot_histogram(counts, fname, xlabel="proposal"):
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    bins = np.arange(0, 7) + 0.5
    plt.figure()
    plt.hist(counts, bins, histtype='bar', facecolor='green', rwidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.savefig(fname + ".png")
    plt.close()


def plot_histograms(counts, fname):
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    bins = np.arange(0, 1, 0.1) + 0.05

    x0 = counts["d_txt"]
    x1 = counts["d_img"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax0, ax1 = axes.flatten()

    ax0.hist(x0, bins, histtype='bar', facecolor='green', rwidth=0.8)
    ax0.set_xlabel('Edit Text')
    ax0.set_ylabel('Counts')

    ax1.hist(x1, bins, histtype='bar', facecolor='green', rwidth=0.8)
    ax1.set_xlabel('Edit Image')

    fig.tight_layout()
    plt.savefig(fname + ".png")
    plt.close()


def simple_plots(xs, ys, path_fig):
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    for k, v in ys.iteritems():
        plt.figure()
        plt.plot(xs, v)
        plt.xlabel("Max Length")
        plt.ylabel(k)
        plt.savefig("_".join([path_fig] + k.split(" ")) + ".png")
        plt.close()