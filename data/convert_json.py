import spacy
from spacy.tokenizer import Tokenizer
import stanza
import json
import argparse
import os
import sys
sys.path.append("../")
from convert_norec_to_sent_graph import *


# def tag_json_files(json_file):
#     for sentence in json_file:
#         tagged_sent = nlp(sentence["text"])
#         conllu = ""
#         for sent in tagged_sent.sentences:
#             for i, token in enumerate(sent.words):
#                 # ID  TOKEN  LEMMA  UPOS  XPOS  MORPH  HEAD_ID  DEPREL
#                 conllu += "{}\t{}\t{}\t{}\t{}\t_\t{}\t{}\t_\t_\n".format(i+1, token.text, token.lemma, token.pos, "_", token.head, token.deprel)
#         sentence["conllu"] = conllu

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab)

def tag_json_files(json_file):
    for sentence in json_file:
        tagged_sent = nlp(sentence["text"])
        conllu = ""
        for i, token in enumerate(tagged_sent.iter_tokens()):
            head = token.head
            conllu += "{}\t{}\t{}\t{}\t{}\t_\t{}\t{}\t_\t_\n".format(i+1, token, token.lemma_, token.pos_, token.tag_, head.i, token.dep_)
        sentence["conllu"] = conllu

def get_sent_conllus(json_file,
                     setup="point_to_root",
                     inside_label=False,
                     use_dep_edges=False,
                     use_dep_labels=False):
    for idx, sentence in enumerate(json_file):
        try:
            sentiment_conllu = ""
            sent_id = sentence["sent_id"]
            text = sentence["text"]
            sentiment_conllu += "# sent_id = {}\n".format(sent_id)
            sentiment_conllu += "# text = {}\n".format(text)
            opinions = sentence["opinions"]
            conllu = sentence["conllu"]
            t2e = tokenidx2edge(conllu)
            t2l = tokenidx2deplabel(conllu)

            if len(opinions) > 0:
                labels = [create_labels(text, o) for o in opinions]
            else:
                labels = [create_labels(text, [])]
            #
            sent_labels = [create_sentiment_dict(l,
                                         setup=setup,
                                         inside_label=inside_label) for l in labels]

            if use_dep_edges:
                if use_dep_labels:
                    sent_labels = [redefine_root_with_dep_edges(s, t2e, t2l) for s in sent_labels]
                else:
                    sent_labels = [redefine_root_with_dep_edges(s, t2e) for s in sent_labels]

            combined_labels = combine_sentiment_dicts(sent_labels)
            conll = create_conll_sent_dict(conllu)
            for i in conll.keys():
                #print(c[i] + "\t" + sd[i])
                sentiment_conllu += conll[i] + "\t" + combined_labels[i] + "\n"
            sentence["sentiment_conllu"] = sentiment_conllu
        except:
            print(idx)

def print_sentconllu(jsonfile, outfile):
    with open(outfile, "w") as o:
        for sent in jsonfile:
            try:
                o.write(sent["sentiment_conllu"] + "\n")
            # does not have a sentiment conllu due to some previous error
            except KeyError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="mpqa")
    parser.add_argument("--out_dir", default="sent_graphs/mpqa")
    parser.add_argument("--setup", default="head_final")
    parser.add_argument("--inside_label", action="store_true")
    parser.add_argument("--use_dep_edges", action="store_true")
    parser.add_argument("--use_dep_labels", action="store_true")

    args = parser.parse_args()

    if args.json_dir == "eu":
        nlp = stanza.Pipeline("eu")

    print("Dataset: {}".format(args.json_dir))
    print("Setup: {}".format(args.setup))
    if args.inside_label:
        print("Using Inside Label")
    if args.use_dep_edges:
        print("Using Dependency Edges to create sentiment graph")
    if args.use_dep_labels:
        print("Using Dependency Labels to create sentiment graph")


    out_dir = os.path.join(args.out_dir, args.setup)
    if args.inside_label:
        out_dir += "-inside_label"
    if args.use_dep_edges:
        out_dir += "-dep_edges"
    if args.use_dep_labels:
        out_dir += "-dep_labels"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.json_dir, "train.json")) as infile:
        train = json.load(infile)
    tag_json_files(train)
    get_sent_conllus(train,
                     setup=args.setup,
                     inside_label=args.inside_label,
                     use_dep_edges=args.use_dep_edges,
                     use_dep_labels=args.use_dep_labels)
    print_sentconllu(train, os.path.join(out_dir, "train.conllu"))

    with open(os.path.join(args.json_dir, "dev.json")) as infile:
        dev = json.load(infile)
    tag_json_files(dev)
    get_sent_conllus(dev,
                     setup=args.setup,
                     inside_label=args.inside_label,
                     use_dep_edges=args.use_dep_edges,
                     use_dep_labels=args.use_dep_labels)
    print_sentconllu(dev, os.path.join(out_dir, "dev.conllu"))

    with open(os.path.join(args.json_dir, "test.json")) as infile:
        test = json.load(infile)
    tag_json_files(test)
    get_sent_conllus(test,
                     setup=args.setup,
                     inside_label=args.inside_label,
                     use_dep_edges=args.use_dep_edges,
                     use_dep_labels=args.use_dep_labels)
    print_sentconllu(test, os.path.join(out_dir, "test.conllu"))
