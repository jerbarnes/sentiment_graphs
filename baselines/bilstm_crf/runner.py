import torch
import re
import os

from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.training.trainer import Params
from allennlp.common.util import import_submodules
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.models.crf_tagger import CrfTagger
from allennlp.predictors import SentenceTaggerPredictor

from torch import optim

from allennlp_reader import Finegrained_Conll_DatasetReader
from allennlp_bilstm_crf import *


def main():
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', action='store', default='configs/bilstm-crf.jsonnet')
    parser.add_argument('--save', action='store', default='results/bilstmcrf')
    parser.add_argument('--num_examples', default=-1, type=int)
    args = parser.parse_args()

    import_submodules("allennlp_bilstm_crf")

    params = Params.from_file(args.config)
    serialization_dir = args.save
    model = train_model(params, serialization_dir, force=True)
