from utils.evaluate import evaluate, evaluate_images_and_edit
from configs.config import Config


config = Config()


path = "results_2/raw.txt"
references = []
hypotheses = []

with open(path) as f:
    hypos = []
    new_example = True
    for idx, line in enumerate(f):
        line = line.strip()
        line = line.split(" ")

        if new_example:
            references.append([line])
            new_example = False
            continue

        if line != ['']:
            hypos.append(line)
        else:
            hypotheses.append(hypos)
            hypos = []
            new_example = True

dummmy_vocab = {word: word for word in config.vocab.iterkeys()}
scores = evaluate(references, hypotheses, dummmy_vocab, "results_2/results.txt", "_END")
print scores

scores2 = evaluate_images_and_edit("results_2/results.txt", "results_2/images/")

print scores
print scores2