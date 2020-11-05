import torch
import h5py
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel
import numpy as np
from tqdm import tqdm
import argparse

import src.col_data as cd

# TODO: need to fix when there's and [UNK] as part of a subword, "``A"   ["[UNK]", "A"]
# sentence representation shape should be (num_tokens, embedding_size)

def check_average_reps(subtokens, bert_reps, tokens):
    subtok = ""
    i = 0
    for j, (tok, rep) in enumerate(zip(subtokens, bert_reps)):
        if "##" in tok:
            subtok += tok[2:]
        else:
            subtok += tok
        if subtok == tokens[i]:
            print(subtok)
            subtok = ""
            i += 1
        elif tok == "[UNK]":
            # Have to account for some cases where the tokenizer breaks a noizy
            # token up and doesn't add ##, such as '”A' -> ["[UNK]", "A"]
            if j < len(subtokens) - 1:
                next_subtok = subtokens[j + 1]
                if next_subtok in tokens[i]:
                    #subtok = tokens[i].replace(next_subtok, "")
                    subtok = subtok[:-5]
                    subtok += '”'
                else:
                    print(subtok)
                    subtok = ""
                    i += 1
            else:
                print(subtok)
                subtok = ""
                i += 1
    return subtok

def average_reps(subtokens, bert_reps, tokens):
    ""
    final_reps = []
    sub_reps = []
    subtok = ""
    i = 0
    for j, (tok, rep) in enumerate(zip(subtokens, bert_reps)):
        if "##" in tok:
            subtok += tok[2:]
        else:
            subtok += tok
        sub_reps.append(rep.detach().numpy())
        if subtok == tokens[i]:
            ave_rep = np.array(sub_reps).mean(axis=0)
            final_reps.append(ave_rep)
            sub_reps = []
            #print(subtok)
            subtok = ""
            i += 1
        elif tok == "[UNK]":
            # Have to account for some cases where the tokenizer breaks a noizy
            # token up and doesn't add ##, such as '”A' -> ["[UNK]", "A"]
            if j < len(subtokens) - 1:
                next_subtok = subtokens[j + 1]
                if next_subtok in tokens[i]:
                    #subtok = tokens[i].replace(next_subtok, "")
                    #subtok = tokens[i][:1]
                    subtok = subtok[:-5]
                    subtok += '”'
                    sub_reps.append(rep.detach().numpy())
                else:
                    ave_rep = np.array(sub_reps).mean(axis=0)
                    final_reps.append(ave_rep)
                    sub_reps = []
                    #print(subtok)
                    subtok = ""
                    i += 1
            else:
                ave_rep = np.array(sub_reps).mean(axis=0)
                final_reps.append(ave_rep)
                sub_reps = []
                #print(subtok)
                subtok = ""
                i += 1
    return np.array(final_reps)

def ee(model, data):
    encodings = []
    sentences = cd.read_col_data(data)
    sids, sents = zip(*[(sent.id, [t.form for t in sent]) for sent in sentences])
    sents = [" ".join(sent) for sent in sents]
    print("Embedding...")
    for sent in tqdm(sents):
        tokens = sent.split()
        tokenized = tokenizer(sent, return_tensors='pt', add_special_tokens=False)
        subtokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
        token_reps, _ = model(**tokenized)
        token_reps = token_reps.squeeze(0)
        assert len(subtokens) == len(token_reps)
        ave_reps = average_reps(subtokens, token_reps, tokens)
        assert len(sent.split()) == len(ave_reps)
        encodings.append(ave_reps)
    return encodings, sids

def write_hdf5(reps, sids, outname):
    print(f"writing to {outname}")
    with h5py.File(f"{outname}", 'w') as fh:
        for x, sid in zip(reps, sids):
            fh.create_dataset(sid, data=[i for i in x], dtype="float32")
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some stuff.')
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--indata", type=str, default=None)
    parser.add_argument("--outdata", type=str, default=None)

    args = parser.parse_args()


    if "roberta" in args.model:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model)
        model = XLMRobertaModel.from_pretrained(args.model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)
        model = BertModel.from_pretrained(args.model)

    reps, sids = ee(model, args.indata)
    write_hdf5(reps, sids, args.outdata)
