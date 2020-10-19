from sklearn.metrics import precision_score, recall_score, f1_score

def binary_tp(gold, pred):
    """
    for each member in pred, if it overlaps with any member of gold,
    return 1
    else
    return 0
    """
    tps = 0
    for p in pred:
        tp = False
        for word in p:
            for span in gold:
                if word in span:
                    tp = True
        if tp is True:
            tps += 1
    return tps

def binary_fn(gold, pred):
    """
    if there is any member of gold that overlaps with no member of pred,
    return 1
    else
    return 0
    """
    fns = 0
    for p in gold:
        fn = True
        for word in p:
            for span in pred:
                if word in span:
                    fn = False
        if fn is True:
            fns += 1
    return fns

def binary_fp(gold, pred):
    """
    if there is any member of pred that overlaps with
    no member of gold, return 1
    else return 0
    """
    fps = 0
    for p in pred:
        fp = True
        for word in p:
            for span in gold:
                if word in span:
                    fp = False
        if fp is True:
            fps += 1
    return fps

def binary_precision(anns, anntype="source"):
    tps = 0
    fps = 0
    for i, ann in anns.items():
        gold = ann["gold"][anntype]
        pred = ann["pred"][anntype]
        tps += binary_tp(gold, pred)
        fps += binary_fp(gold, pred)
    return tps / (tps + fps)


def binary_recall(anns, anntype="source"):
    tps = 0
    fns = 0
    for i, ann in anns.items():
        gold = ann["gold"][anntype]
        pred = ann["pred"][anntype]
        tps += binary_tp(gold, pred)
        fns += binary_fn(gold, pred)
    return tps / (tps + fns)

def binary_f1(anns, anntype="source"):
    prec = binary_precision(anns, anntype)
    rec = binary_recall(anns, anntype)
    return 2 * ((prec * rec) / (prec + rec))



def word2features(sent, i, pos_lex=None, neg_lex=None):
    """
    Creates a feature vector for word i in sent.
    :param sent: a list of word, pos_tag tuples, ex: [("the", 'DET'), ("man",'NOUN'), ("ran",'VERB')]
    :param i   : an integer that refers to the index of the word in the sentence

    This function returns a dictionary object of features of the word at i.

    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        #'word[-5:]': word[-5:],
        #'word[-4:]': word[-4:],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if pos_lex:
        features.update({"word in pos_lex": word in pos_lex})
    if neg_lex:
        features.update({"word in neg_lex": word in neg_lex})

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            #'-1:word[-5:]': word1[-5:],
            #'-1:word[-4:]': word1[-4:],
            #'-1:word[-3:]': word1[-3:],
            #'-1:word[-2:]': word1[-2:],
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
        if pos_lex:
            features.update({"-1:word in pos_lex": word1 in pos_lex})
        if neg_lex:
            features.update({"-1:word in neg_lex": word1 in neg_lex})
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            #'+1:word[-5:]': word1[-5:],
            #'+1:word[-4:]': word1[-4:],
            #'+1:word[-3:]': word1[-3:],
            #'+1:word[-2:]': word1[-2:],
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
        if pos_lex:
            features.update({"+1:word in pos_lex": word1 in pos_lex})
        if neg_lex:
            features.update({"+1:word in neg_lex": word1 in neg_lex})
    else:
        features['EOS'] = True

    return features


def sent2features(sent, pos_lex=None, neg_lex=None):
    """
    Converts a sentence (list of (word, pos_tag) tuples) into a list of feature
    dictionaries
    """
    return [word2features(sent, i, pos_lex, neg_lex) for i in range(len(sent))]

def sent2labels(sent):
    """
    Returns a list of labels, given a list of (token, pos_tag, label) tuples
    """
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    """
    Returns a list of tokens, given a list of (token, pos_tag, label) tuples
    """
    return [token for token, postag, label in sent]

def binary_analysis(pred_analysis):
    print("Binary results:")
    print("#" * 80)
    print()

    # Holders
    prec = binary_precision(pred_analysis, "source")
    rec = binary_recall(pred_analysis, "source")
    f1 = binary_f1(pred_analysis, "source")
    print("Holder prec: {0:.3f}".format(prec))
    print("Holder recall: {0:.3f}".format(rec))
    print("Holder F1: {0:.3f}".format(f1))
    print()

    # Targets
    prec = binary_precision(pred_analysis, "target")
    rec = binary_recall(pred_analysis, "target")
    f1 = binary_f1(pred_analysis, "target")
    print("Target prec: {0:.3f}".format(prec))
    print("Target recall: {0:.3f}".format(rec))
    print("Target F1: {0:.3f}".format(f1))
    print()

    # Polar Expressions
    prec = binary_precision(pred_analysis, "exp")
    rec = binary_recall(pred_analysis, "exp")
    f1 = binary_f1(pred_analysis, "exp")
    print("Polar Exp. prec: {0:.3f}".format(prec))
    print("Polar Exp. recall: {0:.3f}".format(rec))
    print("Polar Exp. F1: {0:.3f}".format(f1))

def proportional_analysis(flat_gold_labels, flat_predictions):
    source_labels = ["B-Source", "I-Source"]
    target_labels = ["B-Target", "I-Target"]
    polar_expression_labels = ["B-Positive", "I-Positive",
                               "B-Negative", "I-Negative"]

    print("Proportional results:")
    print("#" * 80)
    print()

    # Holders
    prec = precision_score(flat_gold_labels, flat_predictions,
                           labels=source_labels, average="micro")
    rec = recall_score(flat_gold_labels, flat_predictions,
                       labels=source_labels, average="micro")
    f1 = f1_score(flat_gold_labels, flat_predictions,
                  labels=source_labels, average="micro")
    print("Holder prec: {0:.3f}".format(prec))
    print("Holder recall: {0:.3f}".format(rec))
    print("Holder F1: {0:.3f}".format(f1))
    print()

    # Targets
    prec = precision_score(flat_gold_labels, flat_predictions,
                           labels=target_labels, average="micro")
    rec = recall_score(flat_gold_labels, flat_predictions,
                       labels=target_labels, average="micro")
    f1 = f1_score(flat_gold_labels, flat_predictions,
                  labels=target_labels, average="micro")
    print("Target prec: {0:.3f}".format(prec))
    print("Target recall: {0:.3f}".format(rec))
    print("Target F1: {0:.3f}".format(f1))
    print()

    # Polar Expressions
    prec = precision_score(flat_gold_labels, flat_predictions,
                           labels=target_labels, average="micro")
    rec = recall_score(flat_gold_labels, flat_predictions,
                       labels=target_labels, average="micro")
    f1 = f1_score(flat_gold_labels, flat_predictions,
                  labels=target_labels, average="micro")
    print("Polar Exp. prec: {0:.3f}".format(prec))
    print("Polar Exp. recall: {0:.3f}".format(rec))
    print("Polar Exp. F1: {0:.3f}".format(f1))
    print()
