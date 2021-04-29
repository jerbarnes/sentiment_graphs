#!/usr/bin/env python
# coding: utf-8

import col_data as cd
import numpy as np
import os
from F1_scorer import (read_labeled,
                       read_unlabeled,
                       get_flat,
                       get_sent_tuples,
                       sent_tuples_in_list,
                       weighted_score,
                       convert_to_targeted
                       )
import time


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


mapping = {'exp-Negative': "exp",
           'exp-negative': "exp",
           'IN:exp-neutral': "exp",
           'exp-neutral': "exp",
           'IN:exp-Negative': "exp",
           'IN:exp-negative': "exp",
           'targ': "targ",
           'exp-positive': "exp",
           'exp-Positive': "exp",
           'IN:exp-Positive': "exp",
           'IN:exp-positive': "exp",
           'holder': "holder",
           'IN:targ': "targ",
           'IN:holder': "holder",
           'IN:exp-None': "exp",
           'exp-None': "exp",
           "exp-conflict": "exp",
           "IN:exp-conflict": "exp",
           "O": "O"}


def span_f1_counts(gold, pred, mapping, test_label="holder"):
    tp, fp, fn = 0, 0, 0
    for gold_sent, pred_sent in zip(gold, pred):
        gold_labels = get_flat(gold_sent)
        pred_labels = get_flat(pred_sent)
        for gold_label, pred_label in zip(gold_labels, pred_labels):
            gold_label = mapping[gold_label]
            pred_label = mapping[pred_label]
            # TP
            if gold_label == pred_label == test_label:
                tp += 1
            # FP
            if gold_label != test_label and pred_label == test_label:
                fp += 1
            # FN
            if gold_label == test_label and pred_label != test_label:
                fn += 1
    return tp, fp, fn


def targeted_f1_counts(gold_edges, pred_edges):
    tp, fp, fn = 0, 0, 0
    #
    for key in gold_edges.keys():
        try:
            gold_targets = convert_to_targeted(gold_edges[key])
            pred_targets = convert_to_targeted(pred_edges[key])
            tp += len(pred_targets.intersection(gold_targets))
            fp += len(pred_targets.difference(gold_targets))
            fn += len(gold_targets.difference(pred_targets))
        except Exception as e:
            print(key, e)
    return tp, fp, fn


def precision_counts(gold, pred):
    """
    True positives / (true positives + false positives)
    """
    tp = 0
    fp = 0
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        for edge_label in p:
            if edge_label in g:
                tp += 1
            else:
                fp += 1
    return tp, fp


def recall_counts(gold, pred):
    """
    True positives / (true positives + false negatives)
    """
    tp = 0
    fn = 0
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        for edge_label in g:
            if edge_label in p:
                tp += 1
            else:
                fn += 1
    return tp, fn


def tuple_precision_counts(gold, pred, keep_polarity=True, weighted=True):
    """
    True positives / (true positives + false positives)
    """
    tp = []
    fp = []
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        ptuples = get_sent_tuples(p)
        gtuples = get_sent_tuples(g)
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    tp.append(weighted_score(stuple, gtuples))
                else:
                    tp.append(1)
            else:
                fp.append(1)
    return sum(tp), sum(fp)


def tuple_recall_counts(gold, pred, keep_polarity=True, weighted=True):
    """
    True positives / (true positives + false negatives)
    """
    tp = []
    fn = []
    #
    assert len(gold) == len(pred)
    #
    for sent_idx in pred.keys():
        p = pred[sent_idx]
        g = gold[sent_idx]
        ptuples = get_sent_tuples(p)
        gtuples = get_sent_tuples(g)
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    tp.append(weighted_score(stuple, ptuples))
                else:
                    tp.append(1)
            else:
                fn.append(1)
    return sum(tp), sum(fn)


# get relevant counts (tp, fp, fn, ...)
# put them in a matrix: row is sentence-id and columns are tp, fp, ...


def read_data(golddir, preddir, setup, r_i):
    goldfile = os.path.join(golddir, setup, "test.conllu")
    gold = list(cd.read_col_data(goldfile))
    lgold = read_labeled(goldfile)
    ugold = read_unlabeled(goldfile)

    L = []

    predfile = os.path.join(preddir, setup, str(r_i), "test.conllu.pred")
    pred = list(cd.read_col_data(predfile))
    lpred = read_labeled(predfile)
    upred = read_unlabeled(predfile)
    # for every sentence
    # for every measure
    # get counts and put them into one long row
    for s_i, s_id in enumerate(lgold):
        assert gold[s_i].id == s_id
        for label in ["holder", "targ", "exp"]:
            tp, fp, fn = span_f1_counts([gold[s_i]], [pred[s_i]],
                                        mapping,
                                        test_label=label
                                        )
            L.extend([tp, fp, fn])
        # targeted f1
        tp, fp, fn = targeted_f1_counts({s_id: lgold[s_id]},
                                        {s_id: lpred[s_id]}
                                        )
        L.extend([tp, fp, fn])
        # UF
        tp, fp = precision_counts({s_id: ugold[s_id]}, {s_id: upred[s_id]})
        L.extend([tp, fp])
        tp, fn = recall_counts({s_id: ugold[s_id]}, {s_id: upred[s_id]})
        L.extend([tp, fn])
        # LF
        tp, fp = precision_counts({s_id: lgold[s_id]}, {s_id: lpred[s_id]})
        L.extend([tp, fp])
        tp, fn = recall_counts({s_id: lgold[s_id]}, {s_id: lpred[s_id]})
        L.extend([tp, fn])
        # USF
        tp, fp = tuple_precision_counts({s_id: lgold[s_id]},
                                        {s_id: lpred[s_id]},
                                        keep_polarity=False,
                                        weighted=True
                                        )
        L.extend([tp, fp])
        tp, fn = tuple_recall_counts({s_id: lgold[s_id]},
                                     {s_id: lpred[s_id]},
                                     keep_polarity=False,
                                     weighted=True
                                     )
        L.extend([tp, fn])
        # LSF
        tp, fp = tuple_precision_counts({s_id: lgold[s_id]},
                                        {s_id: lpred[s_id]},
                                        keep_polarity=True,
                                        weighted=True
                                        )
        L.extend([tp, fp])
        tp, fn = tuple_recall_counts({s_id: lgold[s_id]},
                                     {s_id: lpred[s_id]},
                                     keep_polarity=True,
                                     weighted=True
                                     )
        L.extend([tp, fn])
    return L, len(gold)

# in case one would have tensors and you want to compute prf with the values
# in the last dimension ... works but here : should work as well


def prec(x, i, j):
    return np.divide(x[..., i], (x[..., i] + x[..., j] + 1e-6))


def rec(x, i, j):
    return np.divide(x[..., i], (x[..., i] + x[..., j] + 1e-6))


def fscore(p, r):
    return np.divide(2 * p * r, (p + r + 1e-6))


def compute_scores(scores, counts, p, r, x):
    scores[:, x] = fscore(p, r)
    return scores


def fill_scores(scores, evals, debug=False):
    eval_cnts = range(evals.shape[-1])

    # terrible hardcoded for the 8 measures using 3/4 values to calculate F1

    l_i = 0
    l = -1
    for f_i in range(8):  # there are 8 eval measures
        if f_i < 4:
            # holder 3
            # targ 3
            # exp 3
            # targeted f1 3
            i, j, k = eval_cnts[l_i:l_i + 3]
            p = prec(evals, i, j)
            r = rec(evals, i, k)
            compute_scores(scores, evals, p, r, f_i)
            l_i += 3
        elif f_i >= 4:
            # UF 4
            # LF 4
            # USF 4
            # LSF 4
            i, j, k, l = eval_cnts[l_i:l_i + 4]
            p = prec(evals, i, j)
            r = rec(evals, k, l)
            compute_scores(scores, evals, p, r, f_i)
            l_i += 4
        if debug:
            print(f_i, i, j, k, l)
    return scores


def main(b, golddir, preddir, setup1, r_i1, setup2, r_i2, debug=False):

    if debug:
        s = time.time()

    print(golddir, preddir, setup1, r_i1)
    print(golddir, preddir, setup2, r_i2)
    L1, n = read_data(golddir, preddir, setup1, r_i1)
    L2, _ = read_data(golddir, preddir, setup2, r_i2)

    # number of runs
    b = int(b)

    n_features = int(len(L1) / n)

    if debug:
        print(f"reading in data {time.time() - s}")
        s = time.time()

    M1 = np.array(L1).reshape(int(len(L1) / n_features), n_features)
    M2 = np.array(L2).reshape(int(len(L2) / n_features), n_features)

    # sample 'b' ids for a test set of size 'n' with 'r' runs
    # np.random.choice(n * r, n*b).reshape(b, n)

    if debug:
        print(f"data as matrix {time.time() - s}")
        s = time.time()

    # sample_ids samples b datasets of size n with indices ranging the five
    # runs creating a sample out of all runs
    sample_ids = np.random.choice(n, n*b).reshape(b, n)

    # fill a zero matrix with how often each sentence was drawn in one sample
    samples = np.zeros((n, b))
    for j in range(b):
        for i in range(n):
            samples[sample_ids[j, i], j] += 1

    if debug:
        print(f"get samples {time.time() - s}")
        s = time.time()

    # get the counts for the sample
    # Mx has the counts per sentence and samples chooses how often each sample
    # is taken resulting in a matrix of b rows with sums of tp, fp, fn etc.
    evals1 = (np.einsum('ik,il->lk', M1, samples))
    evals2 = (np.einsum('ik,il->lk', M2, samples))

    if debug:
        print(f"extract sample counts {time.time() - s}")
        s = time.time()

    # compute the eval measures for each row/sample
    sample_scores1 = fill_scores(np.zeros((b, 8)), evals1, False)
    sample_scores2 = fill_scores(np.zeros((b, 8)), evals2, False)

    if debug:
        print(f"compute scores {time.time() - s}")
        s = time.time()

    # scores for the dataset across all runs
    true_scores1 = fill_scores(np.zeros((1, 8)), np.sum(M1, axis=0))
    true_scores2 = fill_scores(np.zeros((1, 8)), np.sum(M2, axis=0))

    # bootstrap scores
    deltas = true_scores1 - true_scores2
    deltas *= 2

    diffs = sample_scores1 - sample_scores2
    diffs_plus = np.where(diffs >= 0, diffs, 0)
    diffs_minus = np.where(diffs < 0, diffs, 0)

    deltas_plus = np.where(deltas > 0, deltas, np.float("inf"))

    deltas_minus = np.where(deltas < 0, deltas, -np.float("inf"))
    s1 = np.sum(diffs_plus > deltas_plus, axis=0)
    s2 = np.sum(diffs_minus < deltas_minus, axis=0)

    if debug:
        print(f"the rest {time.time() - s}")

    if debug:
        print(true_scores1)
        print(true_scores2)

        print(s1 / b)
        print(s2 / b)

        print()
    s1 = s1 / b
    s2 = s2 / b
    end = color.END

    print(f"{color.BOLD}{color.BLUE}{setup1} || {color.RED}{setup2}{color.END}")
    for i, name in enumerate("holder target expression targeted uf lf usf lsf".split()):
        x = true_scores1[0][i]
        y = true_scores2[0][i]
        z = s1[i] if x > y else s2[i]
        if z < 0.05 and x > y:
            bold = color.BLUE
        elif z < 0.05 and y > x:
            bold = color.RED
        else:
            bold = color.END
        print(f"{bold}{name:<13}: {x:.2%}\t{y:.2%}\t{z:.4f}{end}")

    print()


if __name__ == "__main__":
    import sys
    b = 10e4
    golddir = sys.argv[1]
    preddir = sys.argv[2]
    setup1 = sys.argv[3]
    r_i1 = sys.argv[4]
    setup2 = sys.argv[5]
    r_i2 = sys.argv[6]

    main(b, golddir, preddir, setup1, r_i1, setup2, r_i2, debug=False)
