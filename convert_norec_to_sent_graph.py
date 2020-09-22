import norec
import argparse
import os
import json

from nltk.tokenize.simple import SpaceTokenizer
from import_norec import import_conllu

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
            target_tokens = t.split()
            label = "targ"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                tags.append(label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text[0].split()
        label = "targ"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            tags.append(label)
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
            label = "exp-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                tags.append(label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = text[0].split()
        label = "exp-{0}".format(polarity)
        #
        tags = []
        for i, token in enumerate(target_tokens):
            tags.append(label)
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
            label = "holder"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                tags.append(label)
            updates.append((bidx, tags))
        return updates
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = text[0].split()
        label = "holder"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            tags.append(label)
        return [(bidx, tags)]


def replace_with_labels(labels, offsets, bidx, tags):
    try:
        token_idx = offsets.index(bidx)
        for i, tag in enumerate(tags):
            labels[i + token_idx] = tag
        return labels
    except:
        return labels


def create_labels(text, opinion):
    """
    Converts a text (each token separated by a space) and an opinion expression
    into a list of labels for each token in the text.
    """
    offsets = [l[0] for l in tk.span_tokenize(text)]
    #
    labels = ["O"] * len(offsets)
    #
    anns = []
    try:
        anns.extend(get_bio_holder(opinion))
    except:
        pass
    try:
        anns.extend(get_bio_target(opinion))
    except:
        pass
    try:
        anns.extend(get_bio_expression(opinion))
    except:
        pass
    #
    for bidx, tags in anns:
        labels = replace_with_labels(labels, offsets, bidx, tags)
    return labels

def create_sentiment_dict(labels, setup="point_to_root", inside_label=False):
    """
    point_to_root: the final token of the sentiment expression is set as the root and all other labels point to this

    head_first: the first token in the sentiment expression is the root, and the for the holder and target expressions, the first token connects to the root, while the other tokens connect to the first

    head final: the final token in the sentiment expression is the root, and the for the holder and target expressions, the final token connects to the root, while the other tokens connect to the final
    """
    sent_dict = {}
    #
    # associate each label with its token_id
    enum_labels = [(i + 1, l) for i, l in enumerate(labels)]
    #
    if setup in ["point_to_root", "head_final"]:
        enum_labels = list(reversed(enum_labels))
    #
    #for token_id, label in reversed(enum_labels):
    for token_id, label in enum_labels:
        if "exp" in label:
            sent_dict[token_id] = "0:{0}".format(label)
            exp_root_id = token_id
            break
    #
    # point_to_root: point to exp_root_id, regardless of expression type
    if setup == "point_to_root":
        for token_id, label in enum_labels:
            if label == "O":
                sent_dict[token_id] = "_"
            else:
                if token_id not in sent_dict.keys():
                    sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
    # head_first or head_final: first/final point to exp_root, others point inside expression
    else:
        for token_id, label in enum_labels:
            if "targ" in label:
                sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
                targ_root_id = token_id
                break
        #
        for token_id, label in enum_labels:
            if "holder" in label:
                sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
                holder_root_id = token_id
                break
        #
        # set other leafs to point to root
        for token_id, label in enum_labels:
            if label == "O":
                sent_dict[token_id] = "_"
            else:
                if token_id not in sent_dict.keys():
                    if inside_label:
                        label = "IN:" + label
                    if "exp" in label:
                        sent_dict[token_id] = "{0}:{1}".format(exp_root_id, label)
                    elif "targ" in label:
                        sent_dict[token_id] = "{0}:{1}".format(targ_root_id, label)
                    elif "holder" in label:
                        sent_dict[token_id] = "{0}:{1}".format(holder_root_id, label)
    return sent_dict

def create_conll_sent_dict(conllu_sent):
    conll_dict = {}
    for line in conllu_sent.split("\n"):
        if line != "":
            token_id = int(line.split()[0])
            conll_dict[token_id] = line
    return conll_dict

def combine_labels(token_labels):
    final_label = ""
    for l in token_labels:
        if l == "_":
            pass
        else:
            if final_label == "":
                final_label = l
            else:
                final_label += "|" + l
    if final_label == "":
        return "_"
    return final_label


def combine_sentiment_dicts(sentiment_dicts):
    combined = {}
    for i in sentiment_dicts[0].keys():
        labels = [s[i] for s in sentiment_dicts]
        final_label = combine_labels(labels)
        combined[i] = final_label
    return combined


def create_sentiment_conll(finegrained_sent,
                           norec_sents,
                           setup="point_to_root",
                           inside_label=False
                           ):
    sentiment_conll = ""
    #
    sent_id = finegrained_sent["sent_id"]
    text = finegrained_sent["text"]
    opinions = finegrained_sent["opinions"]
    #
    if len(opinions) > 0:
        labels = [create_labels(text, o) for o in opinions]
    else:
        labels = [create_labels(text, [])]
    #
    sent_labels = [create_sentiment_dict(l,
                                         setup=setup,
                                         inside_label=inside_label) for l in labels]
    combined_labels = combine_sentiment_dicts(sent_labels)
    #
    conll = create_conll_sent_dict(norec_sents[sent_id])
    for i in conll.keys():
        #print(c[i] + "\t" + sd[i])
        sentiment_conll += conll[i] + "\t" + combined_labels[i] + "\n"
    return sentiment_conll



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norec_tar", default="data/norec_doc/conllu.tar.gz")
    parser.add_argument("--norec_fine_dir", default="data/norec_fine_final")
    parser.add_argument("--setup", default="head_first")
    parser.add_argument("--inside_label", action="store_true")

    args = parser.parse_args()

    """
    pull in conllu and metada from original NoReC data and
    parse sents and create dictionary where keys are sent-ids
    and values are conllu sentences
    """
    train_sents, dev_sents, test_sents = import_conllu(args.norec_tar)

    # match up fine-grained sentences with their conllu
    with open(os.path.join(args.norec_fine_dir, "train.json")) as infile:
        norec_fine_train = json.load(infile)
    with open(os.path.join(args.norec_fine_dir, "dev.json")) as infile:
        norec_fine_dev = json.load(infile)
    with open(os.path.join(args.norec_fine_dir, "test.json")) as infile:
        norec_fine_test = json.load(infile)

    # take only annotated sentences and covert to sentiment graphs
    train_anns = []
    for s in norec_fine_train:
        try:
            train_anns.append((s["sent_id"], s["text"], create_sentiment_conll(s, train_sents, setup=args.setup, inside_label=args.inside_label)))
        # if the sent_id is not found in the document-level NoReC data
        except KeyError:
            #print(s)
            pass
        # if there is a tokenization error
        except UnboundLocalError:
            #print(s)
            pass

    dev_anns = []
    for s in norec_fine_dev:
        try:
            dev_anns.append((s["sent_id"], s["text"], create_sentiment_conll(s, dev_sents, setup=args.setup, inside_label=args.inside_label)))
        except KeyError:
            pass
        except UnboundLocalError:
            print(s)
            pass

    test_anns = []
    for s in norec_fine_test:
        try:
            test_anns.append((s["sent_id"], s["text"], create_sentiment_conll(s, test_sents, setup=args.setup, inside_label=args.inside_label)))
        except KeyError:
            pass
        except UnboundLocalError:
            print(s)
            pass

    # print the datasets to file
    if args.inside_label:
        outdir = args.setup + "-inside_label"
    else:
        outdir = args.setup
    os.makedirs("data/sent_graphs/{0}".format(outdir), exist_ok=True)

    with open("data/sent_graphs/{0}/train.conllu".format(outdir), "w") as outfile:
        for sent_id, text, sent in train_anns:
            outfile.write("# sent_id = {0}\n".format(sent_id))
            outfile.write("# text = {0}\n".format(text))
            outfile.write(sent + "\n")

    with open("data/sent_graphs/{0}/dev.conllu".format(outdir), "w") as outfile:
        for sent_id, text, sent in dev_anns:
            outfile.write("# sent_id = {0}\n".format(sent_id))
            outfile.write("# text = {0}\n".format(text))
            outfile.write(sent + "\n")

    with open("data/sent_graphs/{0}/test.conllu".format(outdir), "w") as outfile:
        for sent_id, text, sent in test_anns:
            outfile.write("# sent_id = {0}\n".format(sent_id))
            outfile.write("# text = {0}\n".format(text))
            outfile.write(sent + "\n")
