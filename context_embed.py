import torch
import h5py
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse

import src.col_data as cd

def embed(model, data):
    encodings = []
    sentences = cd.read_col_data(data)
    sids, sents = zip(*[(sent.id, [t.form for t in sent]) for sent in sentences])
    print("Embedding...")
    # iterate over sentences
    for sent in tqdm(sents):
        sent_reps = []
        # create an averaged representation for each token
        for token in sent:
            tokenized = tokenizer(token, return_tensors='pt', add_special_tokens=False)
            subtokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
            tok_reps_cl_emb = model(**tokenized)
            # average all subtoken embeddings
            token_reps = tok_reps_cl_emb[0].squeeze(0).mean(0)
            sent_reps.append(token_reps)
        # check that the number of tokens and number of embeddings is equal
        assert len(sent) == len(sent_reps)
        encodings.append(torch.stack(sent_reps).detach().numpy())
    return encodings, sids


def write_hdf5(reps, sids, outname):
    print(f"writing to {outname}")
    with h5py.File(f"{outname}", 'w') as fh:
        for rep, sid in zip(reps, sids):
            fh.create_dataset(sid, data=rep, dtype="float32")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create contextualized embeddings for the data.')
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--indata", type=str, default=None)
    parser.add_argument("--outdata", type=str, default=None)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    reps, sids = embed(model, args.indata)
    write_hdf5(reps, sids, args.outdata)
