# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from collections import defaultdict
import random

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import torchtext

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertModel


class TargetPooledClassification(nn.Module):
    def __init__(self,
                 pretrained_model_dir,
                 hidden_dropout_prob,
                 num_labels,
                 pool_target="cls"
                 ):
        super(TargetPooledClassification, self).__init__()
        self.num_labels = num_labels
        self.pool_target = pool_target
        #
        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        print(self.bert.config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,
                                    num_labels)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        #
        encoded_layers, _ = self.bert(input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids
                                      )
        #
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output.squeeze(0)
        #
        # Get the pooled representation of the embeddings that correspond
        # to the target
        batch_size, sequence_length, embedding_dim = sequence_output.shape
        # target_mask.shape() == (batch_size, sequence_length)
        target_mask = torch.zeros((batch_size, sequence_length))
        # set the target indices to 1, everything else to 0
        target_mask[:, 0] += 1
        target_mask = target_mask.unsqueeze(2) # (batch_size, sequence_length, 1)
        #
        # multiply sequence_output by target_mask to keep only target
        # embeddings and then take their mean pooled representation
        sequence_output = sequence_output * target_mask
        pooled_output = sequence_output.mean(dim=1)
        #
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        #
        outputs = (logits,)
        #
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss
        #
        return outputs

class ABSATokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

    @classmethod
    def _read_torchtext_tabular(cls, input_file):
        """Reads a torchtext tabular dataset"""
        return open_split(input_file, lower_case=False)


def add_span_tags_to_text(tokens, e1, e2, method=1):
    """
    Example:  sent = ['I', 'didn't", 'really', 'like', 'the', 'movie', '.']
              e1 = [0, 0, 0, 1, 0, 0]
              e2 = [0, 0, 0, 0, 1, 1]
              label = 1

     Method 1: replace spans with single entity token
               [CLS] I didn't really [$E1] [$E2]

     Method 2: replace spans with entity tokens for each token in entity
               [CLS] I didn't really [$E1] [$E2] [$E2]

     Method 3: add entity identifiers before and after entity tokens
               [CLS] I didn't really [$E1] like [$E1] [$E2] the movie [$E2]

     Method 4: add full sentence and then entities afterwards separated
              [CLS] I didn't really like the movie [SEP] like [SEP] the movie
    """
    new_sent = []
    if method == 1:
        e1_finished = False
        e2_finished = False
        for token, e1_idx, e2_idx in zip(tokens, e1, e2):
            if e1_idx == 1 and e1_finished == False:
                new_sent.append("[E1]")
                e1_finished = True
            if e2_idx == 1 and e2_finished == False:
                new_sent.append("[E2]")
                e2_finished = True
            if e1_idx == 0 and e2_idx == 0:
                new_sent.append(token)
    if method == 2:
        for token, e1_idx, e2_idx in zip(tokens, e1, e2):
            if e1_idx == 1:
                new_sent.append("[E1]")
            if e2_idx == 1:
                new_sent.append("[E2]")
            if e1_idx == 0 and e2_idx == 0:
                new_sent.append(token)
    if method == 3:
        e1b, e1e = (e1.index(1), len(e1) - 1 - e1[::-1].index(1))
        e2b, e2e = (e2.index(1), len(e2) - 1 - e2[::-1].index(1))
        for i, token in enumerate(tokens):
            if i == e1b:
                new_sent.append("[E1]")
                new_sent.append(token)
                if i == e1e:
                    new_sent.append("[E1]")
            elif i == e1e:
                new_sent.append(token)
                new_sent.append("[E1]")
            elif i == e2b:
                new_sent.append("[E2]")
                new_sent.append(token)
                if i == e2e:
                    new_sent.append("[E2]")
            elif i == e2e:
                new_sent.append(token)
                new_sent.append("[E2]")
            else:
                new_sent.append(token)
    if method == 4:
        e1_tokens = [tokens[i] for i, j in enumerate(e1) if j == 1]
        e2_tokens = [tokens[i] for i, j in enumerate(e2) if j == 1]
        return " ".join(tokens + ["[SEP]"] + e1_tokens + ["[SEP]"] + e2_tokens)
    return " ".join(new_sent)


class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_conll_examples(self, data_file, name):
        """See base class."""
        return self._create_examples(
            self._read_torchtext_tabular(data_file), name)

    def get_train_examples(self, data_dir, fn="train_rels.json", method=1):
        """See base class."""
        return self._create_examples(
            self._read_torchtext_tabular(os.path.join(data_dir, fn)), "train", method=method)

    def get_dev_examples(self, data_dir, fn="dev_rels.json", method=1):
        """See base class."""
        return self._create_examples(
            self._read_torchtext_tabular(os.path.join(data_dir, fn)), "dev", method=method)

    def get_test_examples(self, data_dir, fn="test_rels.json", method=1):
        """See base class."""
        return self._create_examples(
            self._read_torchtext_tabular(os.path.join(data_dir, fn)), "test", method=method)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, method):
        examples = []
        for idx, data in enumerate(lines):
            guid = "%s-%s" % (set_type, idx)
            text = data.text
            e1 = data.e1
            e2 = data.e2
            label = data.label
            text_a = add_span_tags_to_text(text, e1, e2, method=method)
            examples.append(InputExample(guid=guid, text_a=text_a,
                                     text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if mode!="ae":
            tokens_a = tokenizer.tokenize(example.text_a)
        else: #only do subword tokenization.
            tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode!="ae":
            label_id = label_map[example.label]
        else:
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb=[label_map[label] for label in labels_a]
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features

def open_split(data_file, lower_case=False):
    text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
    e1 = torchtext.data.Field(batch_first=True)
    e2 = torchtext.data.Field(batch_first=True)
    label = torchtext.data.Field(sequential=False)
    data = torchtext.data.TabularDataset(data_file, format="json", fields={"text": ("text", text), "e1": ("e1", e1), "e2": ("e2", e2), "label": ("label", label)})
    return data

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
