## Fine-grained Sentiment Analysis as Dependency Graph Parsing

This repository contains the code and datasets described in following paper: [Fine-grained Sentiment Analysis as Dependency Graph Parsing]().

## Problem description

Fine-grained sentiment analysis can be theoretically cast as an information extraction problem in which one attempts to find all of the opinion tuples $O = O_i,\ldots,O_n$ in a text. Each opinion $O_i$ is a tuple $(h, t, e, p)$

where $h$ is a \textbf {holder} who expresses a \textbf{polarity} $p$ towards a \textbf{target} $t$ through a \textbf{sentiment expression} $e$, implicitly defining the relationships between these elements.

The two examples below (first in English, then in Basque) show the conception of *sentiment graphs*.

![multilingual example](./figures/multi_sent_graph.png)

Rather than treating this as a sequence-labeling task, we can treat it as a bilexical dependency graph prediction task, although some decisions must me made. We create two versions (a) *head-first* and (b) *head-final*, shown below:

![bilexical](./figures/bilexical.png)



## Requirements

1. python3
2. pytorch
3. matplotlib
4. sklearn
5. gensim
6. numpy
7. h5py
8. transformers
9. tqdm


## Data collection and preprocessing

We provide the preprocessed bilexical sentiment graph data as conllu files in 'data/sent_graphs'. If you want to run the experiments, you can use this data directly. If, however, you are interested in how we create the data, you can use the following steps.

The first step is to download and preprocess the data, and then create the sentiment dependency graphs. The original data can be downloaded and converted to json files using the scripts found at https://github.com/jerbarnes/finegrained_data. After creating the json files for the finegrained datasets following the instructions, you can then place the directories (renamed to 'mpqa', 'ds_unis', 'norec_fine', 'eu', 'ca') in the 'data' directory.

After that, you can use the available scripts to create the bilexical dependency graphs, as mentioned in the paper.

```
cd data
./create_english_sent_graphs.sh
./create_euca_sent_graphs.sh
./create_norec_sent_graphs
cd ..
```



## Experimental results

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
./scripts/run_base.sh
./scripts/run_bert.sh
```

