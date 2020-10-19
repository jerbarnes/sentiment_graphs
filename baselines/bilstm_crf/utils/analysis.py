from read_data import *
import os
from nltk import FreqDist
from nltk.corpus import stopwords
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

test_files = ['601408', '104051', '303032', '004983', '704794', '000913', '601365', '301662', '202298', '700863', '702352', '002988', '104332', '104219', '501595', '300186', '000787', '400175', '104870', '500201', '302108', '003121', '304196', '703256', '103988', '104097', '104538', '201344', '202063', '000298', '104200', '600639', '304926', '202868', '302975']

train_files = ['107972', '100632', '107011', '703109', '101421', '101419', '700471', '304562', '301835', '101882', '701111', '004068', '301688', '701363', '600051', '201529', '703281', '300972', '111678', '301338', '500127', '100122', '704907', '500043', '001521', '302647', '108264', '100831', '103635', '304002', '109346', '101807', '107326', '304135', '101277', '105567', '102158', '102687', '105855', '101554', '602087', '110965', '100120', '501225', '108139', '003619', '109566', '300746', '105723', '200520', '102688', '110899', '703367', '106155', '001233', '105705', '003382', '102362', '304705', '100392', '001627', '004273', '702788', '003523', '105573', '109371', '202124', '001349', '000076', '102679', '108182', '600117', '600982', '302663', '300040', '700556', '003339', '100866', '202601', '002618', '110840', '703462', '704469', '105177', '102095', '103737', '111035', '003803', '001564', '111170', '702913', '001965', '109458', '107878', '001478', '100915', '705034', '301240', '109901', '003662', '500437', '107563', '705080', '003889', '108638', '002800', '602054', '703046', '704974', '109519', '300474', '601563', '101014', '701716', '600163', '302850', '004340', '304814', '400523', '202043', '106032', '000004', '500921', '101115', '107548', '303715', '600007', '300178', '002747', '104901', '107375', '200607', '201078', '500322', '201174', '302573', '202607', '702187', '600997', '105949', '003939', '701606', '200183', '002051', '105795', '102138', '109173', '700338', '602157', '305017', '702603', '110371', '601158', '300058', '601171', '100183', '102015', '103801', '004230', '602126', '601300', '106602', '700513', '102176', '003283', '001842', '303346', '107610', '109778', '102727', '105977', '702152', '301323', '305169', '103580', '101857', '500984', '304841', '003480', '200099', '302049', '704138', '703816', '001922', '101012', '302159', '704052', '302087', '601962', '107077', '300789', '602002', '702956', '200014', '101381', '001061', '200937', '101562', '004025', '600911', '200247', '600441', '300016', '002380', '701242', '001392', '001756', '703730', '400835', '700726', '106244', '106487', '700917', '102338', '703152', '109617', '110737', '303566', '201911', '102359', '110622', '703512', '106360', '103512', '001885', '300086', '102929', '109100', '301630', '301212']

dev_files = ['500718', '000778', '105373', '501135', '400109', '104926', '400620', '004745', '102329', '602282', '300433', '300002', '601625', '700798', '000335', '705105', '304685', '200102', '304825', '108886', '500716', '301571', '000286', '704010', '108702', '400295', '400215', '000227', '202263', '701679', '104987', '004702', '303164']


def add_to_dist(polarity, intensity):
    x = np.zeros(6)
    if polarity == "Negative":
        if intensity == "Strong":
            x[0] = 1
        elif intensity == "Standard":
            x[1] = 1
        elif intensity == "Slight":
            x[2] = 1
    elif polarity == "Positive":
        if intensity == "Slight":
            x[3] = 1
        elif intensity == "Standard":
            x[4] = 1
        elif intensity == "Strong":
            x[5] = 1
    return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    full_annotations = {}
    targets = []
    sources = []
    expressions = []



    for subdir in os.listdir(args.data_dir):
        if os.path.isdir(os.path.join(args.data_dir, subdir)):
            for file in os.listdir(os.path.join(args.data_dir, subdir)):
                if ".txt" in file:

                    idx = file.split(".")[0].split("_")[0]
                    textfile = os.path.join(args.data_dir, subdir, file)
                    ann_file = textfile[:-4] + ".ann"
                    ann_d, ann = get_anns(ann_file)

                    text = open(textfile).read()
                    full_annotations[idx] = {"Text": text,
                                            "Num_sents": None,
                                            "Sent_lengths": [],
                                           "Anns": {}}

                    for ex in ann.keys():
                        full_annotations[idx]["Anns"][ex] = {}

                        tspans = ann[ex]["Source"]
                        if tspans != []:
                            target = ""
                            for span in tspans:
                                bindx, findx = span.split()
                                bindx = int(bindx)
                                findx = int(findx)
                                target += text[bindx:findx]
                            sources.append(target.lower())
                            full_annotations[idx]["Anns"][ex]["Source"] = target.lower()

                    for ex in ann.keys():
                        tspans = ann[ex]["Target"]
                        if tspans != []:
                            target = ""
                            for span in tspans:
                                bindx, findx = span.split()
                                bindx = int(bindx)
                                findx = int(findx)
                                target += text[bindx:findx]
                            targets.append(target.lower())
                            full_annotations[idx]["Anns"][ex]["Target"] = target.lower()

                    for ex in ann.keys():
                        tspans = ann[ex]["Polar_expression"]
                        pol = ann[ex]["Polarity"]
                        intensity = ann[ex]["Intensity"]
                        if tspans != []:
                            target = ""
                            for span in tspans:
                                bindx, findx = span.split()
                                bindx = int(bindx)
                                findx = int(findx)
                                target += text[bindx:findx]
                            expressions.append(target.lower())
                            full_annotations[idx]["Anns"][ex]["Polar_expression"] = target.lower()
                            full_annotations[idx]["Anns"][ex]["Polarity"] = pol
                            full_annotations[idx]["Anns"][ex]["Intensity"] = intensity

                    sents = re.split("[\n]+", text.strip())
                    num_sents = len(sents)
                    full_annotations[idx]["Num_sents"] = num_sents

                    sent_lengths = []
                    for sent in sents:
                        sent_lengths.append(len(sent.split()))
                    full_annotations[idx]["Sent_lengths"] = sent_lengths

    num_sents = 0
    sent_lengths = []

    num_targets = 0
    targ_lengths = []

    num_source = 0
    source_lengths = []

    num_polar_exp = 0
    polar_exp_lengths = []

    implicit_targets = 0
    implicit_holders = 0


    polarity_distribution = np.zeros(6)


    for doc in full_annotations.values():
        num_sents += doc["Num_sents"]
        sent_lengths.extend(doc["Sent_lengths"])
        for anno in doc["Anns"].values():
            try:
                targ = anno["Target"]
                num_targets += 1
                targ_lengths.append(len(targ.split()))
            except:
                implicit_targets += 1
                pass

            try:
                source = anno["Source"]
                num_source += 1
                source_lengths.append(len(source.split()))
            except:
                implicit_holders += 1
                pass

            try:
                pe = anno["Polar_expression"]
                num_polar_exp += 1
                polar_exp_lengths.append(len(pe.split()))
                pol = anno["Polarity"]
                intensity = anno["Intensity"]
                polarity_distribution += add_to_dist(pol, intensity)
            except:
                pass


    print("Num. Sents: {0}".format(num_sents))
    print("Num. Holders: {0}".format(num_source))
    print("Num. Targets: {0}".format(num_targets))
    print("Num. Polar Exps.: {0}".format(num_polar_exp))
    print()
    print("Avg. Sent Length: {0:.1f}".format(np.mean(sent_lengths)))
    print("Avg. Holder Length: {0:.1f}".format(np.mean(source_lengths)))
    print("Avg. Target Length: {0:.1f}".format(np.mean(targ_lengths)))
    print("Avg. Polar Exp Length: {0:.1f}".format(np.mean(polar_exp_lengths)))
    print()
    print("Implicit Holders: {0}".format(implicit_holders))
    print("Implicit Targets: {0}".format(implicit_targets))


    train_num_sents = 0
    dev_num_sents = 0
    test_num_sents = 0

    train_num_targets = 0
    dev_num_targets = 0
    test_num_targets = 0

    train_num_source = 0
    dev_num_source = 0
    test_num_source = 0

    train_num_polar_exp = 0
    dev_num_polar_exp = 0
    test_num_polar_exp = 0

    for idx, doc in full_annotations.items():
        if idx in train_files:
            train_num_sents += doc["Num_sents"]
            for anno in doc["Anns"].values():
                try:
                    targ = anno["Target"]
                    train_num_targets += 1
                except:
                    pass
                try:
                    source = anno["Source"]
                    train_num_source += 1
                except:
                    pass
                try:
                    pe = anno["Polar_expression"]
                    train_num_polar_exp += 1
                except:
                    pass
        if idx in dev_files:
            dev_num_sents += doc["Num_sents"]
            for anno in doc["Anns"].values():
                try:
                    targ = anno["Target"]
                    dev_num_targets += 1
                except:
                    pass
                try:
                    source = anno["Source"]
                    dev_num_source += 1
                except:
                    pass
                try:
                    pe = anno["Polar_expression"]
                    dev_num_polar_exp += 1
                except:
                    pass
        if idx in test_files:
            test_num_sents += doc["Num_sents"]
            for anno in doc["Anns"].values():
                try:
                    targ = anno["Target"]
                    test_num_targets += 1
                except:
                    pass
                try:
                    source = anno["Source"]
                    test_num_source += 1
                except:
                    pass
                try:
                    pe = anno["Polar_expression"]
                    test_num_polar_exp += 1
                except:
                    pass

    print("Train###########################")
    print("Num. Sents: {0}".format(train_num_sents))
    print("Num. Holders: {0}".format(train_num_source))
    print("Num. Targets: {0}".format(train_num_targets))
    print("Num. Polar Exps.: {0}".format(train_num_polar_exp))
    print()

    print("Dev###########################")
    print("Num. Sents: {0}".format(dev_num_sents))
    print("Num. Holders: {0}".format(dev_num_source))
    print("Num. Targets: {0}".format(dev_num_targets))
    print("Num. Polar Exps.: {0}".format(dev_num_polar_exp))
    print()

    print("Test###########################")
    print("Num. Sents: {0}".format(test_num_sents))
    print("Num. Holders: {0}".format(test_num_source))
    print("Num. Targets: {0}".format(test_num_targets))
    print("Num. Polar Exps.: {0}".format(test_num_polar_exp))
    print()






    if args.plot:
        if args.normalize:
            polarity_distribution /= polarity_distribution.sum()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both',length=0)

        ax.barh(range(len(polarity_distribution)), polarity_distribution, zorder=3)
        ax.set_yticklabels(["", "strong neg.", "neg.", "slight neg.", "slight pos.", "pos.", "strong pos."])


        plt.grid(axis="x", linestyle="--", zorder=0)
        plt.tight_layout()
        plt.show()



    # check overlap between train and test sets

    train = {}
    dev = {}
    test = {}

    for idx, doc in full_annotations.items():
        if idx in train_files:
            train[idx] = doc
        if idx in dev_files:
            dev[idx] = doc
        if idx in test_files:
            test[idx] = doc


    train_holders = []
    dev_holders = []
    test_holders = []

    train_targets = []
    dev_targets = []
    test_targets = []

    train_exp = []
    dev_exp = []
    test_exp = []

    for doc in train.values():
        for anno in doc["Anns"].values():
            try:
                targ = anno["Target"]
                train_targets.append(targ)
            except:
                pass
            try:
                source = anno["Source"]
                train_holders.append(source)
            except:
                pass
            try:
                pe = anno["Polar_expression"]
                train_exp.append(pe)
            except:
                pass

    for doc in dev.values():
        for anno in doc["Anns"].values():
            try:
                targ = anno["Target"]
                dev_targets.append(targ)
            except:
                pass
            try:
                source = anno["Source"]
                dev_holders.append(source)
            except:
                pass
            try:
                pe = anno["Polar_expression"]
                dev_exp.append(pe)
            except:
                pass

    for doc in test.values():
        for anno in doc["Anns"].values():
            try:
                targ = anno["Target"]
                test_targets.append(targ)
            except:
                pass
            try:
                source = anno["Source"]
                test_holders.append(source)
            except:
                pass
            try:
                pe = anno["Polar_expression"]
                test_exp.append(pe)
            except:
                pass

#train_holders = set(train_holders)
#train_targets = set(train_targets)
#train_exp = set(train_exp)

#dev_holders = set(dev_holders)
#dev_targets = set(dev_targets)
#dev_exp = set(dev_exp)

#test_holders = set(test_holders)
#test_targets = set(test_targets)
#test_exp = set(test_exp)

#train_test_holder_overlap = len(train_holders.intersection(test_holders)) / len(train_holders.union(test_holders))

#train_test_target_overlap = len(train_targets.intersection(test_targets)) / len(train_targets.union(test_targets))

#train_test_exp_overlap = len(train_exp.intersection(test_exp)) / len(train_exp.union(test_exp))

train_holders = [l.split() for l in train_holders]
train_holders = [l for s in train_holders for l in s]
dev_holders = [l.split() for l in dev_holders]
dev_holders = [l for s in dev_holders for l in s]
test_holders = [l.split() for l in test_holders]
test_holders = [l for s in test_holders for l in s]


train_targets = [l.split() for l in train_targets]
train_targets = [l for s in train_targets for l in s]
dev_targets = [l.split() for l in dev_targets]
dev_targets = [l for s in dev_targets for l in s]
test_targets = [l.split() for l in test_targets]
test_targets = [l for s in test_targets for l in s]


train_exp = [l.split() for l in train_exp]
train_exp = [l for s in train_exp for l in s]
dev_exp = [l.split() for l in dev_exp]
dev_exp = [l for s in dev_exp for l in s]
test_exp = [l.split() for l in test_exp]
test_exp = [l for s in test_exp for l in s]

def overlap(x, y):
    # make sure longest list is first
    l = list([x, y])
    l.sort(key=len, reverse=True)
    x, y = l

    o = 0
    for i in x:
        if i in y:
            o += 1
    return o

train_test_holder_overlap = overlap(train_holders, test_holders) / (len(train_holders) + len(test_holders))
train_test_target_overlap = overlap(train_targets, test_targets) / (len(train_targets) + len(test_targets))
train_test_exp_overlap = overlap(train_exp, test_exp) / (len(train_exp) + len(test_exp))

print("Train test holder overlap: {0:.3f}".format(train_test_holder_overlap))
print("Train test target overlap: {0:.3f}".format(train_test_target_overlap))
print("Train test exp overlap: {0:.3f}".format(train_test_exp_overlap))

standard_pos = []
standard_neg = []
standard_pos_length = 0
standard_neg_length = 0

strong_pos = []
slight_pos = []
strong_pos_length = 0
strong_neg_length = 0

strong_neg = []
slight_neg = []
slight_pos_length = 0
slight_neg_length = 0

for doc in full_annotations.values():
    for ann in doc["Anns"].values():
        try:
            intensity = ann["Intensity"]
            pol = ann["Polarity"]
            if pol == "Positive":
                if intensity == "Strong":
                    strong_pos.append(ann["Polar_expression"])
                    strong_pos_length += len(ann["Polar_expression"].split())
                if intensity == "Slight":
                    slight_pos.append(ann["Polar_expression"])
                    slight_pos_length += len(ann["Polar_expression"].split())
                if intensity == "Standard":
                    standard_pos.append(ann["Polar_expression"])
                    standard_pos_length += len(ann["Polar_expression"].split())
            if pol == "Negative":
                if intensity == "Strong":
                    strong_neg.append(ann["Polar_expression"])
                    strong_neg_length += len(ann["Polar_expression"].split())
                if intensity == "Slight":
                    slight_neg.append(ann["Polar_expression"])
                    slight_neg_length += len(ann["Polar_expression"].split())
                if intensity == "Standard":
                    standard_neg.append(ann["Polar_expression"])
                    standard_neg_length += len(ann["Polar_expression"].split())
        except:
            pass


standard_pos_length /= len(standard_pos)
standard_neg_length /= len(standard_neg)

strong_pos_length /= len(strong_pos)
strong_neg_length /= len(strong_neg)

slight_pos_length /= len(slight_pos)
slight_neg_length /= len(slight_neg)


print()
print("Standard Pos length: {0:.3f}".format(standard_pos_length))
print("Standard Neg length: {0:.3f}".format(standard_neg_length))

print("Strong Pos length: {0:.3f}".format(strong_pos_length))
print("Strong Neg length: {0:.3f}".format(strong_neg_length))

print("Slight Pos length: {0:.3f}".format(slight_pos_length))
print("Slight Neg length: {0:.3f}".format(slight_neg_length))


# words = stopwords.words("norwegian")

# strong_pos = [l.split() for l in strong_pos]
# strong_pos = [l for s in strong_pos for l in s if l not in words]

# slight_pos = [l.split() for l in slight_pos]
# slight_pos = [l for s in slight_pos for l in s if l not in words]

# strong_neg = [l.split() for l in strong_neg]
# strong_neg = [l for s in strong_neg for l in s if l not in words]

# slight_neg = [l.split() for l in slight_neg]
# slight_neg = [l for s in slight_neg for l in s if l not in words]
