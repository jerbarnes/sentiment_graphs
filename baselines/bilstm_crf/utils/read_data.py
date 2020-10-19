import re
import csv
import os
import argparse

def update_tags_with_range(bindx, findx, tags, tag):
    for i in range(bindx, findx):
        if i == bindx:
            tags[i] = "B-" + tag
        else:
            tags[i] = "I-" + tag

def get_anns(ann_file):
    ann_dict = {}
    anns = csv.reader(open(ann_file), delimiter="\t")
    for ann in anns:
        ann_dict[ann[0]] = ann[1:]

    anns_out = {}

    for ann_idx, value in ann_dict.items():
        # for each sentiment event, get all triggers
        if "E" in ann_idx or "EFINP" in ann_idx:
            # set up subdict for this event
            anns_out[ann_idx] = {"Source": [], "Target": [], "Polar_expression": [], "Polarity": None, "Intensity": None}
            #print(value)
            triggers = value[0].split()
            #print(triggers)
            for trigger in triggers:
                tag, trigger_idx = trigger.split(":")

                t_exp = ann_dict[trigger_idx][0]
                #print(t_exp)
                _ , spans = t_exp.split(" ", maxsplit=1)
                if ";" in spans:
                    spans = spans.split(";")
                    for span in spans:
                        anns_out[ann_idx][tag].append(span)
                else:
                    try:
                        anns_out[ann_idx][tag].append(spans)
                    except KeyError:
                        print(ann_file)
                        print("KeyError: '{0}'".format(tag))
                        break

    # Second pass to get all the attributes for the events
    for idx, value in ann_dict.items():
        if "A" in idx:
            # the ones we're interested in have 3 values
            try:
                tag, ann_idx, label = value[0].split()
                anns_out[ann_idx][tag] = label
            except ValueError:
                pass

    return ann_dict, anns_out

def read_data(data_dir):
    # get tokens and final_tags
    tokenized_sents, final_tags = [], []


    for file in os.listdir(data_dir):
        if ".txt" in file:
            textfile = os.path.join(data_dir, file)
            ann_file = textfile[:-4] + ".ann"

            #print(textfile)

            # open text file
            text = open(textfile).read()

            # get all annotations such that we have spans associated with tags in a dictionary
            ann_dict, anns = get_anns(ann_file)

            # set tags for each offset to 'o'
            blank_tags = ["O"]*len(text)

            # get correct tags
            for ann_idx in anns.keys():
                for span in anns[ann_idx]["Source"]:
                    bindx, findx = span.split()
                    bindx = int(bindx)
                    findx = int(findx)
                    update_tags_with_range(bindx, findx, blank_tags, "Source")
                for span in anns[ann_idx]["Target"]:
                    bindx, findx = span.split()
                    bindx = int(bindx)
                    findx = int(findx)
                    update_tags_with_range(bindx, findx, blank_tags, "Target")
                for span in anns[ann_idx]["Polar_expression"]:
                    bindx, findx = span.split()
                    bindx = int(bindx)
                    findx = int(findx)
                    label = anns[ann_idx]["Polarity"]
                    if label == None:
                        #print(ann_idx)
                        pass
                    try:
                        update_tags_with_range(bindx, findx, blank_tags, label)
                    except:
                        print("----")
                        print("Tagging problem")
                        print(ann_file)
                        print(ann_idx)
                        print("----")


            # get doc with char(tag) format
            doc = "".join(["{0}({1})".format(c, t) for c, t in zip(text, blank_tags)])

            # split into tokens
            #tok_tags = re.split("[ \n](\(.*?\))", doc)
            sents = doc.split("\n")
            tok_tags = [l.split() for l in sents]

            for sent in tok_tags:
                toks = []
                tags = []
                for tok_tag in sent:
                    try:
                        tok = "".join(re.split("[()]", tok_tag)[::2])
                        tag = re.split("[()]", tok_tag)[3]
                        if tok != "":
                            #print("{0}\t{1}".format(tok, tag))
                            toks.append(tok)
                            tags.append(tag)
                    except:
                        pass
                #print()
                if toks != []:
                    tokenized_sents.append(toks)
                    final_tags.append(tags)
    return tokenized_sents, final_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data", help="location of directory with train, dev, test subdirectories.")

    args = parser.parse_args()

    for split in ["train", "dev", "test"]:

        tokenized_sents, final_tags = read_data(os.path.join(args.data_dir, split))

        with open("../../{0}.conll".format(split), "w") as out:
            for sent, tags in zip(tokenized_sents, final_tags):
                for tok, tag in zip(sent, tags):
                    out.write("{0}\t{1}\n".format(tok, tag))
                out.write("\n")
