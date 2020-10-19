
def print_example(pred_analysis, n):
    print(" ".join([str(i) for i in pred_analysis[n]["sent"]]))
    print(pred_analysis[n]["gold"])
    print(pred_analysis[n]["pred"])

def get_analysis(sents, y_pred, y_test):
    pred_analysis = {}

    for i, (sent, pred, gold) in enumerate(zip(sents, y_pred, y_test)):
        target = []
        source = []
        exp = []

        ex = None
        s = None
        t = None

        # Sources
        for j, p in enumerate(pred):
            if "Source" in p:
                if s is None:
                    s = []
                    s.append(sent[j][0])
                else:
                    s.append(sent[j][0])
            else:
                if s is not None:
                    source.append(s)
                    s = None

        # Targets
        for j, p in enumerate(pred):
            if "Target" in p:
                if t is None:
                    t = []
                    t.append(sent[j][0])
                else:
                    t.append(sent[j][0])
            else:
                if t is not None:
                    target.append(t)
                    t = None

        # Polar expressions
        for j, p in enumerate(pred):
            if "Pos" in p or "Neg" in p:
            #if "Pol" in p:
                if ex is None:
                    ex = []
                    ex.append(sent[j][0])
                else:
                    ex.append(sent[j][0])
            else:
                if ex is not None:
                    exp.append(ex)
                    ex = None


        gold_target = []
        gold_source = []
        gold_exp = []

        ex = None
        s = None
        t = None

        # Sources
        for j, p in enumerate(gold):
            if "Source" in p:
                if s is None:
                    s = []
                    s.append(sent[j][0])
                else:
                    s.append(sent[j][0])
            else:
                if s is not None:
                    gold_source.append(s)
                    s = None

        # Targets
        for j, p in enumerate(gold):
            if "Target" in p:
                if t is None:
                    t = []
                    t.append(sent[j][0])
                else:
                    t.append(sent[j][0])
            else:
                if t is not None:
                    gold_target.append(t)
                    t = None

        # Polar expressions
        for j, p in enumerate(gold):
            if "Pos" in p or "Neg" in p:
            #if "Pol" in p:
                if ex is None:
                    ex = []
                    ex.append(sent[j][0])
                else:
                    ex.append(sent[j][0])
            else:
                if ex is not None:
                    gold_exp.append(ex)
                    ex = None

                #gold_exp.append(sent[j][0])

        pred_analysis[i] = {}
        pred_analysis[i]["sent"] = [w[0] for w in sent]
        pred_analysis[i]["gold"] = {}
        pred_analysis[i]["gold"]["source"] = gold_source
        pred_analysis[i]["gold"]["target"] = gold_target
        pred_analysis[i]["gold"]["exp"] = gold_exp
        pred_analysis[i]["pred"] = {}
        pred_analysis[i]["pred"]["source"] = source
        pred_analysis[i]["pred"]["target"] = target
        pred_analysis[i]["pred"]["exp"] = exp

    return pred_analysis
