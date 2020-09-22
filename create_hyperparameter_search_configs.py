import numpy as np
import os
import argparse

config_text = """

[data]
train                         = {}
val                           = {}
#predict_file                  =
external                      = {}
#elmo_train                    = hdf5
#elmo_dev                      = hdf5
#elmo_test                     = hdf5
#load=
target_style                  = scope
#other_target_style            = none
#vocab                         =
#help_style=

[training]
batch_size                    = {}
epochs                        = 100
beta1                         = {:.2f}
beta2                         = {:.2f}
l2                            = {:.2f}

[network_sizes]
hidden_lstm                   = {}
hidden_char_lstm              = {}
layers_lstm                   = {}
dim_mlp                       = {}
dim_embedding                 = 100
dim_char_embedding            = {}
early_stopping                = 0
gcn_layers                    = 2

[network]
pos_style                     = xpos
attention                     = bilinear
model_interpolation           = 0.5
loss_interpolation            = 0.025
lstm_implementation           = drop_connect
char_implementation           = convolved
disable_gradient_clip         = False
unfactorized                  = True
emb_dropout_type              = replace
bridge                        = dpa+

[features]
disable_external              = False
disable_char                  = False
disable_lemma                 = False
disable_pos                   = False
disable_form                  = False
use_elmo                      = True
tree                          = False

[dropout]
dropout_embedding             = {:.2f}
dropout_edge                  = {:.2f}
dropout_label                 = {:.2f}
dropout_main_recurrent        = {:.2f}
dropout_recurrent_char        = {:.2f}
dropout_main_ff               = {:.2f}
dropout_char_ff               = {:.2f}
dropout_char_linear           = {:.2f}

[other]
seed                          = -1
force_cpu                     = False

[output]
quiet                         = True
save_every                    = False
disable_val_eval              = False
enable_train_eval             = False
#dir                           =
"""

external = "/cluster/shared/nlpl/data/vectors/20/58.zip"
train = "data/sent_graphs/head_first-inside_label/train.conllu"
dev = "data/sent_graphs/head_first-inside_label/dev.conllu"

batch_sizes = np.arange(10, 101, 10)
beta1s = [0]
beta2s = [0.95]
l2s = np.arange(0.000000003, 0.03, 0.001)

hidden_lstms = np.arange(50, 401, 10)
hidden_char_lstms = np.arange(50, 201, 10)
layers_lstms = [1, 2, 3, 4, 5]
dim_mlps = np.arange(50, 401, 50)
dim_char_embeddings = np.arange(50, 151, 10)

dropout_embeddings = np.arange(0.05, 0.41, 0.01)
dropout_edges = np.arange(0.05, 0.41, 0.01)
dropout_labels = np.arange(0.05, 0.41, 0.01)
dropout_main_recurrents = np.arange(0.05, 0.41, 0.01)
dropout_recurrent_chars = np.arange(0.05, 0.41, 0.01)
dropout_main_ffs = np.arange(0.05, 0.41, 0.01)
dropout_char_ffs = np.arange(0.05, 0.41, 0.01)
dropout_char_linears = np.arange(0.05, 0.41, 0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default=50, type=int)
    args = parser.parse_args()

    for i in range(args.budget):
        # Randomly select
        batch_size = batch_sizes[np.random.randint(len(batch_sizes))]
        beta1 = beta1s[np.random.randint(len(beta1s))]
        beta2 = beta2s[np.random.randint(len(beta2s))]
        l2 = l2s[np.random.randint(len(l2s))]
        hidden_lstm = hidden_lstms[np.random.randint(len(hidden_lstms))]
        hidden_char_lstm = hidden_char_lstms[np.random.randint(len(hidden_char_lstms))]
        layer_lstm = layers_lstms[np.random.randint(len(layers_lstms))]

        dim_mlp = dim_mlps[np.random.randint(len(dim_mlps))]
        dim_char_embedding = dim_char_embeddings[np.random.randint(len(dim_char_embeddings))]

        dropout_embedding = dropout_embeddings[np.random.randint(len(dropout_embeddings))]
        dropout_edge = dropout_edges[np.random.randint(len(dropout_edges))]
        dropout_label = dropout_labels[np.random.randint(len(dropout_labels))]
        dropout_main_recurrent = dropout_main_recurrents[np.random.randint(len(dropout_main_recurrents))]
        dropout_recurrent_char = dropout_recurrent_chars[np.random.randint(len(dropout_recurrent_chars))]
        dropout_main_ff = dropout_main_ffs[np.random.randint(len(dropout_main_ffs))]
        dropout_char_ff = dropout_char_ffs[np.random.randint(len(dropout_char_ffs))]
        dropout_char_linear = dropout_char_linears[np.random.randint(len(dropout_char_linears))]

        new_config = config_text.format(train,
                                        dev,
                                        external,
                                        batch_size,
                                        beta1,
                                        beta2,
                                        l2,
                                        hidden_lstm,
                                        hidden_char_lstm,
                                        layer_lstm,
                                        dim_mlp,
                                        dim_char_embedding,
                                        dropout_embedding,
                                        dropout_edge,
                                        dropout_label,
                                        dropout_main_recurrent,
                                        dropout_recurrent_char,
                                        dropout_main_ff,
                                        dropout_char_ff,
                                        dropout_char_linear)

        os.makedirs("configs/hyperparameter_search", exist_ok=True)
        with open(os.path.join("configs/hyperparameter_search", "config_{}.cfg".format(i)), "w") as outfile:
            outfile.write(new_config)
