import json
import nltk
import re
import argparse
from nltk.tokenize.simple import SpaceTokenizer
import os

tk = SpaceTokenizer()


def get_bio_target(opinion):
    try:
        text, idxs = opinion["Target"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["Polarity"]
            target_tokens = t.split()
            label = "-targ-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = text[0].split()
        label = "-targ-{0}".format(polarity)
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]

def get_bio_expression(opinion):
    try:
        text, idxs = opinion["Polar_expression"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["Polarity"]
            target_tokens = t.split()
            label = "-exp-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = text[0].split()
        label = "-exp-{0}".format(polarity)
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]

def get_bio_holder(opinion):
    try:
        text, idxs = opinion["Source"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            target_tokens = t.split()
            label = "-holder"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text[0].split()
        label = "-holder"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]


def replace_with_labels(labels, offsets, bidx, tags):
    # There are some annotations that missed token level (left out a leading character) that we need to fix
    try:
        token_idx = offsets.index(bidx)
        for i, tag in enumerate(tags):
            labels[i + token_idx] = tag
        return labels
    except:
        return labels

def restart_orphans(labels):
    """Wen opinion expression tags are written on top of previous expression tags,
    I-tags can be orphaned, so they do not correspond with the previous tag. We reset these to a B

        labels : list(Str) tag sequence for a sentence.
    """
    prev = "O"
    for tag_idx,tag in enumerate(labels):
        if tag[0] == "I":
            if prev == "O" or (len(prev)>1 and tag[1:] != prev[1:]):
                labels[tag_idx] = "B"+tag[1:] #Replace I with B since contents is different from prev
                #print("correcting", prev, tag)
        prev = labels[tag_idx]
    return labels


def create_bio_labels(text, opinions):
    offsets = [l[0] for l in tk.span_tokenize(text)]
    #
    columns = ["holder", "target", "expression"]
    labels = {c:["O"] * len(offsets) for c in columns}
    #
    anns = {c:[] for c in columns}


    # TODO: deal with targets which can have multiple polarities, due to
    # contrasting polar expressions. At present the last polarity wins.
    for o in opinions:
        try:
            anns["holder"].extend(get_bio_holder(o))
        except:
            pass
        try:
            anns["target"].extend(get_bio_target(o))
        except:
            pass
        try:
            anns["expression"].extend(get_bio_expression(o))
        except:
            pass
    #
    for c in columns:
        for bidx, tags in anns[c]:
            labels[c] = replace_with_labels(labels[c], offsets, bidx, tags)
        labels[c] = restart_orphans(labels[c])
    return labels


def to_bio(dataset):
    tokenized = []
    all_labels = []
    for i, sent in enumerate(dataset):
        #print(i)
        text = sent["text"]
        opinions = sent["opinions"]
        tokens = text.split()
        labels = create_bio_labels(text, opinions)
        tokenized.append(tokens)
        all_labels.append(labels)
    return tokenized, all_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data/norec_fine_final")
    parser.add_argument("--output_dir", default="data/norec")
    parser.add_argument("--lower", action="store_true")

    args = parser.parse_args()

    sents = []

    for split in ["train", "dev", "test"]:
        with open(os.path.join(args.data_dir, "{0}.json".format(split))) as o:
            data = json.load(o)

        tokenized, labels = to_bio(data)
        if args.lower:
            tokenized = [l.lower() for l in tokenized]
        sents.extend(tokenized)

        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
        with open(os.path.join(args.output_dir, split, "sentence.txt"), "w") as outfile:
            for line in tokenized:
                outfile.write(" ".join(line) + "\n")

        with open(os.path.join(args.output_dir, split, "target.txt"), "w") as outfile:
            for line in labels:
                rep = ["1" if "B" in l else "2" if "I" in l else "0" for l in line["target"]]
                outfile.write(" ".join(rep) + "\n")

        with open(os.path.join(args.output_dir, split, "opinion.txt"), "w") as outfile:
            for line in labels:
                rep = ["1" if "B" in l else "2" if "I" in l else "0" for l in line["expression"]]
                outfile.write(" ".join(rep) + "\n")

        with open(os.path.join(args.output_dir, split, "target_polarity.txt"), "w") as outfile:
            for line in labels:
                rep = ["1" if "Positive" in l else "2" if "Negative" in l else "0" for l in line["target"]]
                outfile.write(" ".join(rep) + "\n")

    fd = nltk.FreqDist()
    for line in sents:
        fd.update(line)

    w2idx = {}
    w2idx['<pad>'] = 0
    for w, _ in fd.most_common():
        w2idx[w.lower()] = len(w2idx)

    with open(os.path.join(args.output_dir, "word2id.txt"), "w") as outfile:
        outfile.write(str(w2idx))
