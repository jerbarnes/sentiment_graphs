import argparse

def get_sent_tuples(labeled_edges, keep_polarity=True):
    sent_tuples = []
    polarities = []
    expressions = []
    targets = []
    holders = []
    for token_idx, edge, label in labeled_edges:
        if edge == "0":
            polarity = label.split("-")[-1]
            polarities.append(polarity)
            exp = [token_idx]
            for t_idx, e, l in labeled_edges:
                if e == token_idx and polarity in l:
                    exp.append(t_idx)
            expressions.append(exp)
    for token_idx, edge, label in labeled_edges:
        if label == "targ":
            exp_idx = edge
            target = [token_idx]
            for t_idx, e, l in labeled_edges:
                if e in target:
                    target.append(t_idx)
            targets.append((exp_idx, target))
    for token_idx, edge, label in labeled_edges:
        if label == "holder":
            exp_idx = edge
            holder = [token_idx]
            for t_idx, e, l in labeled_edges:
                if e in holder:
                    holder.append(t_idx)
            holders.append((exp_idx, holder))
    for exp, pol in zip(expressions, polarities):
        current_targets = [t for idx, t in targets if idx == exp[0]]
        current_holders = [t for idx, t in holders if idx == exp[0]]
        if current_targets == []:
            current_targets = [[]]
        if current_holders == []:
            current_holders = [[]]
        for target in current_targets:
            for holder in current_holders:
                sent_tuples.append((frozenset(holder), frozenset(target), frozenset(exp), pol))
    return list(set(sent_tuples))

def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(exp1.intersection(exp2)) > 0:
            if keep_polarity:
                if pol1 == pol2:
                    #print(holder1, target1, exp1, pol1)
                    #print(holder2, target2, exp2, pol2)
                    return True
            else:
                #print(holder1, target1, exp1, pol1)
                #print(holder2, target2, exp2, pol2)
                return True
    return False

def weighted_tuples_precision(sent_tuple1, list_of_sent_tuples):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(exp1.intersection(exp2)) > 0:
                holder_overlap = len(holder1.intersection(holder2)) / len(holder1)
                target_overlap = len(target1.intersection(target2)) / len(target1)
                exp_overlap = len(exp1.intersection(exp2)) / len(exp1)
                return (holder_overlap + target_overlap + exp_overlap) / 3
    return 0

def weighted_score(sent_tuple1, list_of_sent_tuples):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(exp1.intersection(exp2)) > 0:
                holder_overlap = len(holder1.intersection(holder2)) / len(holder2)
                target_overlap = len(target1.intersection(target2)) / len(target2)
                exp_overlap = len(exp1.intersection(exp2)) / len(exp2)
                return (holder_overlap + target_overlap + exp_overlap) / 3
    return 0

def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
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
    return tp, fp, sum(tp) / (sum(tp) + sum(fp))

def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
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
    return tp, fn, sum(tp) / (sum(tp) + sum(fn))

def tuple_F1(gold, pred, keep_polarity=True, weighted=True):
    tp, fp, prec = tuple_precision(gold, pred, keep_polarity, weighted)
    tp, fn, rec = tuple_recall(gold, pred, keep_polarity, weighted)
    return 2 * (prec * rec) / (prec + rec)

def read_labeled(file):
    """
    Read in dependency edges and labels as tuples
    (token_idx, dep_idx, label)
    """
    labeled_edges = {}
    sent_id = None
    sent_edges = []
    for line in open(file):
        if line.startswith("# sent_id"):
            sent_id = line.strip().split(" = ")[-1]
        if line.strip() == "" and sent_id is not None:
            labeled_edges[sent_id] = sent_edges
            sent_edges = []
            sent_id = None
        if line[0].isdigit():
            split = line.strip().split("\t")
            idx = split[0]
            edge_label = split[-1]
            #print(edge_label)
            if edge_label is not "_":
                if "|" in edge_label:
                    for el in edge_label.split("|"):
                        edge, label = el.split(":", 1)
                        sent_edges.append((idx, edge, label))
                else:
                    edge, label = edge_label.split(":", 1)
                    sent_edges.append((idx, edge, label))
    return labeled_edges


def read_unlabeled(file):
    """
    Read in dependency edges as tuples
    (token_idx, dep_idx)
    """
    unlabeled_edges = {}
    sent_id = None
    sent_edges = []
    for line in open(file):
        if line.startswith("# sent_id"):
            sent_id = line.strip().split(" = ")[-1]
        if line.strip() == "" and sent_id is not None:
            unlabeled_edges[sent_id] = sent_edges
            sent_edges = []
            sent_id = None
        if line[0].isdigit():
            split = line.strip().split("\t")
            idx = split[0]
            edge_label = split[-1]
            #print(edge_label)
            if edge_label is not "_":
                if "|" in edge_label:
                    for el in edge_label.split("|"):
                        edge, label = el.split(":", 1)
                        sent_edges.append((idx, edge))
                else:
                    edge, label = edge_label.split(":", 1)
                    sent_edges.append((idx, edge))
    return unlabeled_edges

def precision(gold, pred):
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
    return tp / (tp + fp)

def recall(gold, pred):
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
    return tp / (tp + fn)

def F1(gold, pred):
    prec = precision(gold, pred)
    rec = recall(gold, pred)
    return 2 * (prec * rec) / (prec + rec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("goldfile")
    parser.add_argument("predfile")

    args = parser.parse_args()


    lgold = read_labeled(args.goldfile)
    lpred = read_labeled(args.predfile)

    ugold = read_unlabeled(args.goldfile)
    upred = read_unlabeled(args.predfile)

    print("Unlabeled")
    prec = precision(ugold, upred)
    rec = recall(ugold, upred)
    f1 = F1(ugold, upred)
    print("P: {0:.3f}".format(prec))
    print("R: {0:.3f}".format(rec))
    print("F1: {0:.3f}".format(f1))
    print()

    print("Labeled")
    prec = precision(lgold, lpred)
    rec = recall(lgold, lpred)
    f1 = F1(lgold, lpred)
    print("P: {0:.3f}".format(prec))
    print("R: {0:.3f}".format(rec))
    print("F1: {0:.3f}".format(f1))
    print()

    print("Sentiment Tuple - Polarity ")
    tp, fp, prec = tuple_precision(lgold, lpred, False)
    tp, fn, rec = tuple_recall(lgold, lpred, False)
    f1 = tuple_F1(lgold, lpred, False)
    print("P: {0:.3f}".format(prec))
    print("R: {0:.3f}".format(rec))
    print("F1: {0:.3f}".format(f1))
    print()

    print("Sentiment Tuple + Polarity")
    tp, fp, prec = tuple_precision(lgold, lpred)
    tp, fn, rec = tuple_recall(lgold, lpred)
    f1 = tuple_F1(lgold, lpred)
    print("P: {0:.3f}".format(prec))
    print("R: {0:.3f}".format(rec))
    print("F1: {0:.3f}".format(f1))
