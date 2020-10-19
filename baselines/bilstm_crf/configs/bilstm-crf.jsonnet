{
  "dataset_reader": {
    "type": "fine-grained",
  },
  "train_data_path": "/home/jeremy/Exps/OLD/norec_fine/experiments/train.conll",
  "validation_data_path": "/home/jeremy/Exps/OLD/norec_fine/experiments/dev.conll",
"model": {
    "type": "crf_tagger",
    "label_encoding": "BIO",
    "dropout": 0.3,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "/home/jeremy/Exps/embeddings/norwegian/model.txt",
            "embedding_dim": 100,
            "trainable": true
        },
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 300,
      "num_layers": 1,
      "dropout": 0.5,
      "bidirectional": true
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 40,
    "patience": 5,
    "cuda_device": -1
  }
}
