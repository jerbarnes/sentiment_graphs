import tarfile
import numpy as np
import pandas as pd


def import_conllu(compressed_corpus):
    #
    train_sents = {}
    dev_sents = {}
    test_sents = {}
    #
    # open compressed corpus
    tar = tarfile.open(compressed_corpus, "r:gz")
    #
    sent = ""
    sent_id = ""
    #
    # for each file in the corpus
    for fname in tar.getmembers():
        if ".conllu" in fname.path:
            # content is the conllu file in tar
            content = tar.extractfile(fname)
            # go through each line
            for line in content:
                # decode to get
                line = line.decode("utf8")
                if line == "\n":
                    if len(sent) > 0:
                        if "train" in fname.path:
                            train_sents[sent_id] = sent
                        elif "dev" in fname.path:
                            dev_sents[sent_id] = sent
                        elif "test" in fname.path:
                            test_sents[sent_id] = sent
                        sent = ""
                        sent_id = ""
                elif line.startswith("# sent_id ="):
                    # set the sent_id 'document_id-paragraph_id-sent_id'
                    sent_id = line.strip().split(" = ")[-1]
                elif line.startswith("#"):
                    pass
                else:
                    sent += line
    #
    return train_sents, dev_sents, test_sents
