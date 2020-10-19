from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.fields import TextField, LabelField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from overrides import overrides

from allennlp.data.dataset_readers import DatasetReader


import re
import csv
import os

@DatasetReader.register('fine-grained')
class Finegrained_Conll_DatasetReader(DatasetReader):
    def __init__(self, num_examples: int = -1) -> None:
        super().__init__()
        self._tokenizer = JustSpacesWordSplitter()
        self._indexer = {'tokens': SingleIdTokenIndexer('tokens')}
        self._num_examples = num_examples

    @overrides
    def _read(self, file_dir: str):
        tokens = []
        labels = []
        for line in open(file_dir):

            if not line.rstrip("\n"):
                yield self.text_to_instance(tokens, labels)
                tokens = []
                labels = []

            else:
                try:
                    token, label = line.strip().split("\t")
                    tokens.append(Token(token))
                    labels.append(label)
                except:
                    pass
                    #print(line)


    @overrides
    def text_to_instance(self, tokens, labels) -> Instance:
        fields = {}

        sequence = TextField(tokens, self._indexer)
        fields['tokens'] = sequence
        fields['tags'] = SequenceLabelField(labels, sequence, "labels")

        return Instance(fields)

