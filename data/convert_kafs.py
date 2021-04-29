from lxml import etree
from lxml.etree import fromstring
import os
import json
import re
import json
import argparse
import sys
import stanza

sys.path.append("../")
from convert_norec_to_sent_graph import *

from nltk.tokenize import WhitespaceTokenizer

xmlparser = etree.XMLParser(recover=True, encoding='utf8')

catalan_tags_to_upos = {"A": "ADJ",
                        "N": "NOUN",
                        "CC": "CCONJ",
                        "CS": "SCONJ",
                        "D": "DET",
                        "P": "PRON",
                        "R": "ADV",
                        "S": "ADP",
                        "VA": "AUX",
                        "VM": "VERB",
                        "VS": "VERB",
                        "Z": "NUM",
                        "W": "X",
                        "I": "INTJ",
                        "F": "PUNCT"
                        }

euskera_tags_to_upos = {"ADJ": "ADJ",
                        "IZE": "NOUN",
                        "LOT_JNT": "CCONJ",
                        "LOT_MEN": "SCONJ",
                        "LOT_LOK": "ADV",
                        "ERL": "SCONJ",
                        "BA_BA_BA": "SCONJ",
                        "AREAGOATU_AREAGOATU_AREAGOATU": "NOUN",
                        "DET": "DET",
                        "IOR": "PRON",
                        "ADB": "ADV",
                        "PRT": "ADV",
                        "ITJ": "INTJ",
                        "HAOS": "AUX",
                        "ADL": "AUX",
                        "ADT": "AUX",
                        "ADI": "VERB",
                        "BST": "X",
                        "PUNCT": "PUNCT",
                        "O": "PUNCT"
                        }

def map_to_upos(morphfeat, map):
    for lang_pos, upos in map.items():
        if morphfeat.startswith(lang_pos):
            return upos

def get_sents(sents):
    flipped = {}
    for x, i in sents.items():
        if i in flipped:
            flipped[i].append(x)
        else:
            flipped[i] = [x]
    return flipped

def get_parsed_sent(xml_file, sent_num, map, nlp):
    catalan = False
    conllu = ""
    mark_xml = open(xml_file).read().encode('utf8')
    base_root = fromstring(mark_xml, xmlparser)
    #
    tokens = {}
    sents = {}
    terms = {}
    for annotation in base_root:
        if annotation.tag == "text":
            for token in annotation:
                token_idx = token.get("id")
                if token_idx is None:
                    catalan = True
                    token_idx = token.get("wid")
                tok = token.text
                sent = token.get("sent")
                tokens[token_idx] = tok
                sents[token_idx] = sent
        if annotation.tag == "terms":
            for term in annotation:
                if term.tag == "term":
                    idx = term.get("id")
                    if idx is None:
                        idx = term.get("tid")
                    #print(idx)
                    idx = idx.replace("t", "w")
                    lemma = term.get("lemma")
                    if catalan:
                        pos = term.get("pos")
                    else:
                        pos = term.get("morphofeat")
                    upos = map_to_upos(pos, map)
                    terms[idx] = (lemma, pos, upos)
   # if catalan is True:
   #     print("Processing Catalan Kafs")
   # else:
   #     print("Processing Basque Kafs")
    sentidx2tokenidx = get_sents(sents)
    token_idxs = sentidx2tokenidx[str(sent_num)]
    text = " ".join([tokens[i] for i in token_idxs])
    processed = nlp(text)
    deps = [(t.head, t.deprel) for t in processed.sentences[0].words]
    #
    for i, tokenidx in enumerate(token_idxs):
        token = tokens[tokenidx]
        lemma, tag, pos = terms[tokenidx]
        head, deprel = deps[i]
        # ID  TOKEN  LEMMA  UPOS  XPOS  MORPH  HEAD_ID  DEPREL
        conllu += "{}\t{}\t{}\t{}\t{}\t_\t{}\t{}\t_\t_\n".format(i + 1, token, lemma, pos, tag, head, deprel)
    return conllu

def get_sent_conllus(json_file,
                     setup="point_to_root",
                     inside_label=False,
                     use_dep_edges=False,
                     use_dep_labels=False):
    for idx, sentence in enumerate(json_file):
        sentiment_conllu = ""
        sent_id = sentence["sent_id"]
        kaf, sent_num = sent_id.rsplit("-", 1)
        if "/ca/" in kaf:
            catalan = True
            basename = os.path.basename(kaf)
            kaf = "ca/kafs/" + basename + ".kaf"
            pos_map = catalan_tags_to_upos
        else:
            catalan = False
            basename = os.path.basename(kaf)
            kaf = "eu/kafs/" + basename + ".kaf"
            pos_map = euskera_tags_to_upos
        text = sentence["text"]
        sentiment_conllu += "# sent_id = {}\n".format(sent_id)
        sentiment_conllu += "# text = {}\n".format(text)
        opinions = sentence["opinions"]
        conllu = get_parsed_sent(kaf, sent_num, pos_map, nlp)
        t2e = tokenidx2edge(conllu)
        t2l = tokenidx2deplabel(conllu)
        if len(opinions) > 0:
            labels = [create_labels(text, o) for o in opinions]
        else:
            labels = [create_labels(text, [])]

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
    parser.add_argument("--json_dir", default="eu")
    parser.add_argument("--out_dir", default="sent_graphs/eu")
    parser.add_argument("--setup", default="head_final")
    parser.add_argument("--inside_label", action="store_true")
    parser.add_argument("--use_dep_edges", action="store_true")
    parser.add_argument("--use_dep_labels", action="store_true")


    args = parser.parse_args()

    print("Dataset: {}".format(args.json_dir))
    print("Setup: {}".format(args.setup))
    if args.inside_label:
        print("Using Inside Label")
    if args.use_dep_edges:
        print("Using Dependency Edges to create sentiment graph")
    if args.use_dep_labels:
        print("Using Dependency Labels to create sentiment graph")


    if args.json_dir == "eu":
        nlp = stanza.Pipeline(lang='eu', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)
    elif args.json_dir == "ca":
        nlp = stanza.Pipeline("ca", processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)


    out_dir = os.path.join(args.out_dir, args.setup)
    if args.inside_label:
        out_dir += "-inside_label"
    if args.use_dep_edges:
        out_dir += "-dep_edges"
    if args.use_dep_labels:
        out_dir += "-dep_labels"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.json_dir, args.setup), exist_ok=True)

    with open(os.path.join(args.json_dir, "train.json")) as infile:
        train = json.load(infile)
    get_sent_conllus(train,
                     setup=args.setup,
                     inside_label=args.inside_label,
                     use_dep_edges=args.use_dep_edges,
                     use_dep_labels=args.use_dep_labels)
    print_sentconllu(train, os.path.join(out_dir, "train.conllu"))

    with open(os.path.join(args.json_dir, "dev.json")) as infile:
        dev = json.load(infile)
    get_sent_conllus(dev,
                     setup=args.setup,
                     inside_label=args.inside_label,
                     use_dep_edges=args.use_dep_edges,
                     use_dep_labels=args.use_dep_labels)
    print_sentconllu(dev, os.path.join(out_dir, "dev.conllu"))

    with open(os.path.join(args.json_dir, "test.json")) as infile:
        test = json.load(infile)
    get_sent_conllus(test,
                     setup=args.setup,
                     inside_label=args.inside_label,
                     use_dep_edges=args.use_dep_edges,
                     use_dep_labels=args.use_dep_labels)
    print_sentconllu(test, os.path.join(out_dir, "test.conllu"))
