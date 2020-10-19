import torch
import re
import os
import numpy as np

from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.training.trainer import Params
from allennlp.common.util import import_submodules
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive


from allennlp_reader import Finegrained_Conll_DatasetReader
from allennlp_bilstm_crf import *


from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.analyze_predictions import get_analysis
from utils.utils import *


def main():
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lang', action='store')
    parser.add_argument('--config', action='store', default='configs/simple_tagger.jsonnet')
    parser.add_argument('--save', action='store', default='results/bilstmcrf')
    parser.add_argument('--dataset', default="../test.conll")
    args = parser.parse_args()

    import_submodules("allennlp_reader")
    import_submodules("allennlp_bilstm_crf")

    model = load_archive(os.path.join(args.save, "model.tar.gz")).model
    idx2label = model.vocab.get_index_to_token_vocabulary("labels")

    reader = Finegrained_Conll_DatasetReader()

    p = Predictor(model, reader)

    # Keep the sentences, predictions and gold labels by sentence
    sents = []
    y_pred = []
    y_test = []

    # Flatten the predictions and gold labels for sklearn metrics
    flat_predictions = []
    flat_gold_labels = []

    for i in reader.read(args.dataset):
        pred = p.predict_instance(i)['tags']
        gold = [l for l in i["tags"]]
        tokens = [l for l in i["tokens"]]

        # For get_analysis we need each sentence to be a list of tuples
        # where each tuple is (token, pos_tag, gold_tag). We just insert dummy
        # pos tags in this case
        sents.append(list(zip(tokens, ["NONE"] * len(tokens), gold)))
        y_pred.append(pred)
        y_test.append(gold)

        flat_predictions.extend(pred)
        flat_gold_labels.extend(gold)

    # Binary Results
    pred_analysis = get_analysis(sents, y_pred, y_test)

    proportional_analysis(flat_gold_labels, flat_predictions)

    binary_analysis(pred_analysis)
