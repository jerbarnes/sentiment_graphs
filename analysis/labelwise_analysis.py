import argparse
import sys
sys.path.append("../src")
import col_data as cd
from tabulate import tabulate
from collections import Counter
import os

def get_labels(scopes):
    # given a list of scopes, [(14, 'targ'), (18, 'targ')]
    # return a set of the labels set(['targ', 'targ'])
    if scopes == []:
        return set(["O"])
    return set([i[1] for i in scopes])

def labelwise_f1(gold, pred, map_labels=False):
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold_sent, pred_sent in zip(gold, pred):
        for gold_token, pred_token in zip(gold_sent.tokens, pred_sent.tokens):
            gold_labels = get_labels(gold_token.scope)
            pred_labels = get_labels(pred_token.scope)
            if map_labels:
                gold_labels = [mapping[l] for l in gold_labels]
                pred_labels = [mapping[l] for l in pred_labels]
            # True Positive
            for label in pred_labels:
                if label in gold_labels:
                    tp[label] += 1
            #False Positive
                else:
                    fp[label] += 1
            #False Negative
            for label in gold_labels:
                if label not in pred_labels:
                    fn[label] += 1
    all_labels = set(tp.keys()).union(set(fp.keys())).union(set(fn.keys()))
    prec = {}
    rec = {}
    f1 = {}
    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label] + 1e-6)
        r = tp[label] / (tp[label] + fn[label] + 1e-6)
        prec[label] = p
        rec[label] = r
        f1[label] = 2 * p * r / (p + r + 1e-6)
    return tp, fp, fn, prec, rec, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("golddir")
    parser.add_argument("preddir")
    parser.add_argument("--map", default=False, action="store_true")

    args = parser.parse_args()
    print(args.map)

    goldfile = os.path.join(args.golddir, "test.conllu")
    predfile = os.path.join(args.preddir, "test.conllu.pred")

    gold = list(cd.read_col_data(goldfile))
    pred = list(cd.read_col_data(predfile))


    mapping = {'exp-Negative': "exp", 'exp-negative': "exp", 'IN:exp-neutral': "exp", 'exp-neutral': "exp", 'IN:exp-Negative': "exp", 'IN:exp-negative': "exp", 'targ': "targ", 'exp-positive': "exp", 'exp-Positive': "exp", 'IN:exp-Positive': "exp", 'IN:exp-positive': "exp", 'holder': "holder", 'IN:targ': "targ", 'IN:holder': "holder", 'IN:exp-None': "exp", 'exp-None': "exp", "exp-conflict": "exp", "IN:exp-conflict": "exp", "O": "O"}

    tp, fp, fn, prec, rec, f1 = labelwise_f1(gold, pred, args.map)

    prectable = []
    for k, v in sorted(prec.items()):
        prectable.append((k, v*100, rec[k] * 100, f1[k] * 100, tp[k], fp[k], fn[k]))
    print(tabulate(prectable, headers=["Precision", "Recall", "F1", "True Pos.", "False Pos.", "False Neg."], floatfmt=".1f"))
