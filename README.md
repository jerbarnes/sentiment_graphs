## Sentiment Graphs
==============

The main code is in src.



# Requirements

1. python3
2. pytorch
3. matplotlib
4. sklearn
5. gensim
6. numpy
7. h5py
8. transformers
9. tqdm


# Data collection and preprocessing

The first step is to download and preprocess the data, and then create the sentiment dependency graphs. The original data can be downloaded and converted to json files using the scripts found at https://github.com/jerbarnes/finegrained_data. After creating the json files for the finegrained datasets following the instructions, you can then place the directories (renamed to 'mpqa', 'ds_unis', 'norec_fine', 'eu', 'ca') in the 'data' directory.

After that, you can use the available scripts to create the bilexical dependency graphs, as mentioned in the paper.

```
cd data
./create_english_sent_graphs.sh
./create_euca_sent_graphs.sh
./create_norec_sent_graphs
cd ..
```



# Experimental results

To reproduce the results, first you will need to download the word vectors used:

```
mkdir vectors
cd vectors
wget http://vectors.nlpl.eu/repository/20/58.zip
wget http://vectors.nlpl.eu/repository/20/32.zip
wget http://vectors.nlpl.eu/repository/20/34.zip
wget http://vectors.nlpl.eu/repository/20/18.zip
cd ..
```

You will similarly need to extract mBERT token representations for all datasets.
```
./do_bert.sh
```

Finally, you can run the SLURM scripts to reproduce the experimental results.

```
./scripts/run_SLURM_all_BERT.sh
./scripts/run_SLURM_no_BERT.sh
```

