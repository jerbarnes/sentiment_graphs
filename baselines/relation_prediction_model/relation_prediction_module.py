import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
import torchtext
import string
import os
from nltk.tokenize.simple import SpaceTokenizer
from convert_to_bio import get_bio_holder, get_bio_target, get_bio_expression, replace_with_labels
import itertools
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from WordVecs import WordVecs
import numpy as np
import pickle


def create_bio_labels(text, opinion):
    tk = SpaceTokenizer()
    #
    offsets = [l[0] for l in tk.span_tokenize(text)]
    #
    columns = ["Source", "Target", "Polar_expression"]
    labels = {c:["O"] * len(offsets) for c in columns}
    #
    anns = {c:[] for c in columns}
    # TODO: deal with targets which can have multiple polarities, due to
    # contrasting polar expressions. At present the last polarity wins.
    try:
        anns["Source"].extend(get_bio_holder(opinion))
    except:
        pass
    try:
        anns["Target"].extend(get_bio_target(opinion))
    except:
        pass
    try:
        anns["Polar_expression"].extend(get_bio_expression(opinion))
    except:
        pass
    #
    for c in columns:
        for bidx, tags in anns[c]:
            labels[c] = replace_with_labels(labels[c], offsets, bidx, tags)
    return labels

def convert_to_train_example(text, opinions):
    examples = []
    if len(opinions) == 0:
        return []
    elif len(opinions) > 1:
        olabels = []
        ols = []
        for opinion in opinions:
            labels = create_bio_labels(text, opinion)
            ls = {}
            for c in ["Source", "Target", "Polar_expression"]:
                labels[c] = [0 if l == "O" else 1 for l in labels[c]]
                ls[c] = len(set(labels[c]))
            olabels.append(labels)
            ols.append(ls)
            if ls["Target"] > 1:
                examples.append((text, labels["Target"], labels["Polar_expression"], 1))
            if ls["Source"] > 1:
                examples.append((text, labels["Source"], labels["Polar_expression"], 1))
        # Iterate over possible combinations
        x = range(len(olabels))
        for idx1, idx2 in itertools.combinations(x, 2):
            if ols[idx1]["Target"] > 1:
                if olabels[idx1]["Target"] != olabels[idx2]["Target"]:
                    examples.append((text, olabels[idx1]["Target"], olabels[idx2]["Polar_expression"], 0))
            if ols[idx1]["Source"] > 1:
                if olabels[idx1]["Source"] != olabels[idx2]["Source"]:
                    examples.append((text, olabels[idx1]["Source"], olabels[idx2]["Polar_expression"], 0))
            if ols[idx2]["Target"] > 1:
                if olabels[idx1]["Target"] != olabels[idx2]["Target"]:
                    examples.append((text, olabels[idx2]["Target"], olabels[idx1]["Polar_expression"], 0))
            if ols[idx2]["Source"] > 1:
                if olabels[idx1]["Source"] != olabels[idx2]["Source"]:
                    examples.append((text, olabels[idx2]["Source"], olabels[idx1]["Polar_expression"], 0))
    else:
        labels = create_bio_labels(text, opinions[0])
        # convert to boolean and
        ls = {}
        for c in ["Source", "Target", "Polar_expression"]:
            labels[c] = [0 if l == "O" else 1 for l in labels[c]]
            ls[c] = len(set(labels[c]))
        # check if more than one set of labels and if so, add to examples
        if ls["Target"] > 1:
            examples.append((text, labels["Target"], labels["Polar_expression"], 1))
        if ls["Source"] > 1:
            examples.append((text, labels["Source"], labels["Polar_expression"], 1))
    return examples


class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_padded_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)

        words = pack_sequence([w for w, e1, e2, label in batch])
        e1s = pack_sequence([e1 for w, e1, e2, label in batch])
        e2s = pack_sequence([e2 for w, e1, e2, label in batch])
        targets = default_collate([t for w, e1, e2, t in batch])

        return words, e1s, e2s, targets

def create_relation_jsons(filename, outfile):
    data = []
    with open(filename) as infile:
        for sent in json.load(infile):
            data.extend(convert_to_train_example(sent["text"], sent["opinions"]))
    json_data = []
    for text, e1, e2, label in data:
        json_data.append({"text": text, "e1": e1, "e2": e2, "label": label})
    with open(outfile, "w") as o:
        for d in json_data:
            json.dump(d, o)
            o.write("\n")


class Dataset(object):
    def __init__(self, vocab, lower_case, data_dir="../data/datasets/en/sst-fine"):

        self.vocab = vocab
        self.splits = {}

        for name in ["train_rels", "dev_rels", "test_rels"]:
            filename = os.path.join(data_dir, name) + ".json"
            self.splits[name] = self.open_split(filename, lower_case)

        self.labels = [0, 1]

    def open_split(self, data_file, lower_case):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        e1 = torchtext.data.Field(batch_first=True)
        e2 = torchtext.data.Field(batch_first=True)
        label = torchtext.data.Field(sequential=False)
        data = torchtext.data.TabularDataset(data_file, format="json", fields={"text": ("text", text), "e1": ("e1", e1), "e2": ("e2", e2), "label": ("label", label)})
        data_split = [(torch.LongTensor(self.vocab.ws2ids(item.text)),
                       torch.LongTensor(item.e1),
                       torch.LongTensor(item.e2),
                       torch.LongTensor([int(item.label)])) for item in data]
        return data_split

    def get_split(self, name):
        return Split(self.splits[name])

class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda: len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]
        self.idx2w = self.update_idx2w()

    def update_idx2w(self):
        self.idx2w = dict([(i, w) for w, i in self.items()])

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return torch.tensor([self[w] for w in ws], dtype=torch.long)
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        #idxs = set(idx2w.keys())
        #return [idx2w[int(i)] if int(i) in idxs else "UNK" for i in ids]
        return [self.idx2w[int(i)] for i in ids]

class Relation_Model(nn.Module):
    def __init__(self, word2idx,
                 embedding_dim,
                 hidden_dim,
                 embedding_matrix=None,
                 pooling="max",
                 lstm_dropout=0.2,
                 word_dropout=0.4,
                 train_embeddings=False):
        super(Relation_Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2idx)
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.pooling = pooling

        if embedding_matrix is not None:
            weight = torch.FloatTensor(embedding_matrix)
            self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=False)
            self.word_embeds.requires_grad = train_embeddings
        else:
            self.word_embeds = nn.Embedding(len(word2idx), embedding_dim)
        self.criterion = nn.BCELoss()

        self.e1_embeds = nn.Embedding(2, embedding_dim)
        self.e2_embeds = nn.Embedding(2, embedding_dim)

        self.sent_lstm = nn.LSTM(embedding_dim,
                                 hidden_dim,
                                 num_layers=1,
                                 bidirectional=True)

        self.e1_lstm = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=1,
                               bidirectional=True)

        self.e2_lstm = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=1,
                               bidirectional=True)


        # Maps the output of the LSTM into tag space.
        self.sigmoid = nn.Sigmoid()
        self.word_dropout = nn.Dropout(word_dropout)
        self.ff = nn.Linear(hidden_dim * 6, 1)

    def init_hidden1(self, batch_size=1):
        h0 = torch.zeros((self.sent_lstm.num_layers*(1+self.sent_lstm.bidirectional),
                                  batch_size, self.sent_lstm.hidden_size))
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(self, sent, e1, e2):
        batch_size = sent.batch_sizes[0]
        emb = self.word_embeds(sent.data)
        emb = self.word_dropout(emb)
        packed_emb = PackedSequence(emb, sent.batch_sizes)
        self.hidden = self.init_hidden1(batch_size)
        output, (hn, cn) = self.sent_lstm(packed_emb, self.hidden)
        #text_rep = hn.reshape(batch_size, self.hidden_dim * 2)
        o, _ = pad_packed_sequence(output, batch_first=True)
        if self.pooling == "max":
            text_rep, _ = o.max(dim=1)
        else:
            text_rep = o.mean(dim=1)

        emb = self.e1_embeds(e1.data)
        packed_emb = PackedSequence(emb, e1.batch_sizes)
        self.hidden = self.init_hidden1(batch_size)
        output, (hn, cn) = self.e1_lstm(packed_emb, self.hidden)
        #e1_rep = hn.reshape(batch_size, self.hidden_dim * 2)
        o, _ = pad_packed_sequence(output, batch_first=True)
        if self.pooling == "max":
            e1_rep, _ = o.max(dim=1)
        else:
            e1_rep = o.mean(dim=1)

        emb = self.e2_embeds(e2.data)
        packed_emb = PackedSequence(emb, e2.batch_sizes)
        self.hidden = self.init_hidden1(batch_size)
        output, (hn, cn) = self.e2_lstm(packed_emb, self.hidden)
        #e2_rep = hn.reshape(batch_size, self.hidden_dim * 2)
        o, _ = pad_packed_sequence(output, batch_first=True)
        if self.pooling == "max":
            e2_rep, _ = o.max(dim=1)
        else:
            e2_rep = o.mean(dim=1)

        conc = torch.cat((text_rep, e1_rep, e2_rep), dim=1)

        pred = self.ff(conc)

        return self.sigmoid(pred)

    def predict(self, dataloader):
        preds = []
        self.eval()
        for sent, e1, e2, label in tqdm(dataloader):
            pred = self.forward(sent, e1, e2)
            preds.extend(pred)
        pred_labels = [1 if i > .5 else 0 for i in preds]
        self.train()
        return pred_labels, preds

    def test_model(self, dataloader):
        preds, golds = [], []
        self.eval()
        batch_loss = 0
        batches = 0
        for sent, e1, e2, label in tqdm(dataloader):
            batches += 1
            pred = rel_model(sent, e1, e2)
            preds.extend(pred)
            golds.extend(label)
            loss = self.criterion(pred, label.float())
            batch_loss += loss.data
        pred_labels = [1 if i > .5 else 0 for i in preds]
        golds = [int(i) for i in golds]
        f1 = f1_score(pred_labels, golds, average="macro")
        full_loss = batch_loss / batches
        self.train()
        return f1, full_loss


def distance(sent, e1, e2):
    e1_idxs = [sent.index(e1[0]), sent.index(e1[-1])]
    e2_idxs = [sent.index(e2[0]), sent.index(e2[-1])]
    m = abs(max(e1_idxs) - min(e2_idxs))
    n = abs(max(e2_idxs) - min(e1_idxs))
    return min((m, n))


if __name__ == "__main__":

    EMBEDDINGS = "../../../embeddings/norwegian/model.txt"
    DATASET = "norec_fine"
    NUMBER_OF_RUNS = 1
    random_seeds = [123, 456, 789, 101112, 131415]
    LEARNING_RATE = 0.001
    POOLING = "max"

    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    print("loading embeddings from {0}".format(EMBEDDINGS))
    embeddings = WordVecs(EMBEDDINGS)
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 1
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    dataset = Dataset(vocab, True, "../../data/norec_fine_final")
    train = dataset.get_split("train_rels")
    dev = dataset.get_split("dev_rels")
    test = dataset.get_split("test_rels")

    # For development, only use small amount of training data
    train.data = train.data[:1000]


    # Get new embedding matrix so that words not included in pretrained embeddings have a random embedding

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, 100))
    new_embeddings = np.zeros((diff, 100))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))

    # create relation prediction model
    rel_model = Relation_Model(vocab,
                               embedding_dim=100,
                               hidden_dim=50,
                               embedding_matrix=new_matrix,
                               pooling=POOLING)


    train_loader = DataLoader(train,
                              batch_size=20,
                              collate_fn=train.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev,
                            batch_size=20,
                            collate_fn=train.collate_fn,
                            shuffle=False)

    test_loader = DataLoader(test,
                             batch_size=10,
                             collate_fn=test.collate_fn,
                             shuffle=False)

    sentiment_optimizer = torch.optim.Adam(rel_model.parameters(),
                                           lr=LEARNING_RATE)


    # for i, run in enumerate(range(NUMBER_OF_RUNS)):

    #     # Save the model parameters
    #     param_file = (dict(vocab.items()),
    #                   new_matrix.shape)

    #     basedir = os.path.join("saved_models",
    #                            "{0}".format(DATASET))
    #     outfile = os.path.join(basedir,
    #                            "params.pkl")
    #     print("Saving model parameters to " + outfile)
    #     os.makedirs(basedir, exist_ok=True)

    #     with open(outfile, "wb") as out:
    #         pickle.dump(param_file, out)

    #     print("RUN {0}".format(run + 1))
    #     best_dev_f1 = 0.0

    #     # set random seed for reproducibility
    #     np.random.seed(random_seeds[i])
    #     torch.manual_seed(random_seeds[i])


    #     for epoch in range(10):
    #         rel_model.train()
    #         batch_loss = 0
    #         batches = 0
    #         preds, golds = [], []
    #         for sent, e1, e2, label in tqdm(train_loader, desc="epoch: {0}".format(epoch + 1 )):
    #             rel_model.zero_grad()
    #             pred = rel_model(sent, e1, e2)
    #             preds.extend(pred)
    #             golds.extend(label)
    #             loss = rel_model.criterion(pred, label.float())
    #             batch_loss += loss.data
    #             batches += 1
    #             loss.backward()
    #             sentiment_optimizer.step()
    #         pred_labels = [1 if i > .5 else 0 for i in preds]
    #         golds = [int(i) for i in golds]
    #         acc = accuracy_score(pred_labels, golds)
    #         f1 = f1_score(pred_labels, golds, average="macro")

    #         print("Train Loss: {0:.3f}".format(batch_loss / batches))
    #         print("Train f1: {0:.3f}".format(f1))

    #         f1, loss = rel_model.test_model(dev_loader)

    #         print("Dev f1: {0:.3f}".format(f1))

    #         if f1 > best_dev_f1:
    #                 best_dev_f1 = f1
    #                 print("NEW BEST DEV F1: {0:.3f}".format(f1))


    #                 basedir = os.path.join("saved_models",
    #                                        "{0}".format(DATASET),
    #                                        "{0}".format(run + 1))
    #                 outname = "epochs:{0}-lstm_dim:{1}-lstm_layers:{2}-lr:{3}-pooling:{4}-devf1:{5:.3f}".format(epoch + 1, rel_model.sent_lstm.hidden_size, rel_model.sent_lstm.num_layers, LEARNING_RATE, POOLING, f1)
    #                 modelfile = os.path.join(basedir,
    #                                          outname)
    #                 os.makedirs(basedir, exist_ok=True)
    #                 print("saving model to {0}".format(modelfile))
    #                 torch.save(rel_model.state_dict(), modelfile)

    print("loading best model...")
    rel_model.load_state_dict(torch.load("saved_models/norec_fine/1/epochs:5-lstm_dim:50-lstm_layers:1-lr:0.001-devf1:0.646"))

    vocab.update_idx2w()
    sents, e1s, e2s, labels = [], [], [], []
    for sent, e1, e2, label in test_loader:
        o, _ = pad_packed_sequence(sent, batch_first=True)
        sents.extend([vocab.ids2sent(s) for s in o])
        o, _ = pad_packed_sequence(e1, batch_first=True)
        e1s.extend(o)
        o, _ = pad_packed_sequence(e2, batch_first=True)
        e2s.extend(o)
        labels.extend([int(i) for i in label])

    e1s2 = [[sents[i][j] for j, l in enumerate(e1s[i]) if l == 1] for i in range(len(sents))]
    e2s2 = [[sents[i][j] for j, l in enumerate(e2s[i]) if l == 1] for i in range(len(sents))]

    pred_labels, preds = rel_model.predict(test_loader)

    mistakes = []
    mistake_distances = []
    correct = []
    correct_distances = []

    for sent, e1, e2, label, pred in zip(sents, e1s2, e2s2, labels, pred_labels):
        if label == pred:
            correct.append((sent, e1, e2, label))
            correct_distances.append(distance(sent, e1, e2))
        else:
            mistakes.append((sent, e1, e2, label))
            mistake_distances.append(distance(sent, e1, e2))

