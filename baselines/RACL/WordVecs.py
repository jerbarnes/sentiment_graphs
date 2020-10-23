import numpy as np
import pickle
from scipy.spatial.distance import cosine
import gensim

if __name__ == "__main__":

    #embedding_file = "norec-sgns-100-window8-negative5.txt"
    embedding_file = "58/model.bin"

    with open("data/norec/word2id.txt", 'r', encoding='utf-8') as f:
        word_dict = eval(f.read())

    vocab = list(word_dict.keys())

    b = False
    if "bin" in embedding_file:
        b = True

    vecs = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=b)

    vec_dim = vecs.vector_size

    embs = np.zeros((len(vocab), vec_dim))

    for i, word in enumerate(vocab):
        if word in vecs:
            embs[i] = vecs[word]

    if "norec" in embedding_file:
        outfile = "data/norec/domain_embedding.npy"
    else:
        outfile = "data/norec/glove_embedding.npy"
    with open(outfile, "wb") as f:
        np.save(f, embs)
