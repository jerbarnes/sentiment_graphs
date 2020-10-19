import sys
import os
from utils.read_data import read_data
from utils.utils import *
from utils.analyze_predictions import get_analysis

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from nltk import PerceptronTagger
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
#from sklearn.cross_validation import KFold
from ufal.udpipe import Model, Pipeline, ProcessingError
import numpy as np



def BOW(sent, w2idx):
    """
    Creates a bag of words representation of a sentence
    given a word, index dictionary
    """
    vec = np.zeros(len(w2idx))
    for w in sent:
        try:
            vec[w2idx[w]] = 1
        except KeyError:
            vec[0] = 1
    return vec

def get_vocab(expressions):
    vocab = {}
    vocab['UNK'] = 0
    for exp, label in expressions:
        for w in exp:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

if __name__ == '__main__':

    data_dir = "../data"

    train_sents, train_tags = read_data(os.path.join(data_dir, "train"))
    dev_sents, dev_tags = read_data(os.path.join(data_dir, "dev"))
    test_sents, test_tags = read_data(os.path.join(data_dir, "test"))

    #Load tokenizer
    model = Model.load("../norwegian-bokmaal-ud-2.4-190531.udpipe")
    pipeline = Pipeline(model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    error = ProcessingError()

    # Read whole input
    train_text = "\n".join([" ".join(s) for s in train_sents])
    dev_text = "\n".join([" ".join(s) for s in dev_sents])
    test_text = "\n".join([" ".join(s) for s in test_sents])

    # Process data
    print("preprocessing corpus...")
    train_processed = pipeline.process(train_text, error)
    dev_processed = pipeline.process(dev_text, error)
    test_processed = pipeline.process(test_text, error)

    train_pos_tags = []
    dev_pos_tags = []
    test_pos_tags = []

    pos = []
    for line in train_processed.splitlines():
        if line == "":
            train_pos_tags.append(pos)
            pos = []
            continue
        if line[0] == '#':
            continue
        else:
            pos.append(line.split("\t")[3])

    pos = []
    for line in dev_processed.splitlines():
        if line == "":
            dev_pos_tags.append(pos)
            pos = []
            continue
        if line[0] == '#':
            continue
        else:
            pos.append(line.split("\t")[3])

    pos = []
    for line in test_processed.splitlines():
        if line == "":
            test_pos_tags.append(pos)
            pos = []
            continue
        if line[0] == '#':
            continue
        else:
            pos.append(line.split("\t")[3])

    labels = {'I-Negative', '', 'B-Positive', 'I-Positive', 'I-Target', 'B-Negative', 'B-Source', 'I-Source', 'o', 'B-Target'}

    train_tuples = []
    dev_tuples = []
    test_tuples = []

    for sent, tags, labs in zip(train_sents, train_pos_tags, train_tags):
        s = []
        for w, t, y in zip(sent, tags, labs):
            s.append((w, t, y))
        train_tuples.append(s)

    for sent, tags, labs in zip(dev_sents, dev_pos_tags, dev_tags):
        s = []
        for w, t, y in zip(sent, tags, labs):
            s.append((w, t, y))
        dev_tuples.append(s)

    for sent, tags, labs in zip(test_sents, test_pos_tags, test_tags):
        s = []
        for w, t, y in zip(sent, tags, labs):
            s.append((w, t, y))
        test_tuples.append(s)


    pos = [l.strip() for l in open("../lexicon/positive-words.txt")]
    neg = [l.strip() for l in open("../lexicon/negative-words.txt")]


    X_train = [sent2features(s, pos, neg) for s in train_tuples]
    y_train = [sent2labels(s) for s in train_tuples]

    X_test = [sent2features(s, pos, neg) for s in test_tuples]
    y_test = [sent2labels(s) for s in test_tuples]

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.01, c2=0.1, max_iterations=100, all_possible_transitions=True)

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    gold = [f for i in y_test for f in i]
    pred = [f for i in y_pred for f in i]

    source_labels = ["B-Source", "I-Source"]
    target_labels = ["B-Target", "I-Target"]
    polar_expression_labels = ["B-Positive", "I-Positive",
                               "B-Negative", "I-Negative"]
    #polar_expression_labels = ["B-Pol", "I-Pol"]

    # BINARY Results
    pred_analysis = get_analysis(test_sents, y_pred, y_test)

    proportional_analysis(gold, pred)

    binary_analysis(pred_analysis)

    # ANALYSIS

    target_lengths = []
    source_lengths = []
    expression_lengths = []

    gold_target_lengths = []
    gold_source_lengths = []
    gold_expression_lengths = []

    for i in pred_analysis.keys():
        if pred_analysis[i]["gold"]["source"] != []:
            gold_source_lengths.append(len(pred_analysis[i]["gold"]["source"]))
        if pred_analysis[i]["gold"]["target"] != []:
            gold_target_lengths.append(len(pred_analysis[i]["gold"]["target"]))
        if pred_analysis[i]["gold"]["exp"] != []:
            gold_expression_lengths.append(len(pred_analysis[i]["gold"]["exp"]))

        if pred_analysis[i]["pred"]["source"] != []:
            source_lengths.append(len(pred_analysis[i]["pred"]["source"]))
        if pred_analysis[i]["pred"]["target"] != []:
            target_lengths.append(len(pred_analysis[i]["pred"]["target"]))
        if pred_analysis[i]["pred"]["exp"] != []:
            expression_lengths.append(len(pred_analysis[i]["pred"]["exp"]))

    target_lengths = np.array(target_lengths)
    source_lengths = np.array(source_lengths)
    expression_lengths = np.array(expression_lengths)

    gold_target_lengths = np.array(gold_target_lengths)
    gold_source_lengths = np.array(gold_source_lengths)
    gold_expression_lengths = np.array(gold_expression_lengths)
