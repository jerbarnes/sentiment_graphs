import torch
import h5py
from ELMoForManyLangs.elmoformanylangs import Embedder
import src.col_data as cd 
import argparse

def ee(model, data):
    e = Embedder(model)
    sentences = cd.read_col_data(data)
    sids, sents = zip(*[(sent.id, [t.form for t in sent]) for sent in sentences])
    print("Embedding...")
    elmos = e.sents2elmo(sents)
    return elmos, sids

def write_hdf5(elmos, sids, outname):
    print(f"writing to {outname}")
    with h5py.File(f"{outname}", 'w') as fh:
        for x, sid in zip(elmos, sids):
            fh.create_dataset(sid, data=[i for i in x], dtype="float32")
    print("Done")

def get_args():
    parser = argparse.ArgumentParser(description='Process some stuff.')
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--indata", type=str, default=None)
    parser.add_argument("--outdata", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    write_hdf5(*ee(args.model, args.indata), args.outdata)
    
