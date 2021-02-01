Sentiment Graphs
==============

The main code is in src.

First create the sentiment graphs
---

Move the data to the 'code' directory.


```
cd data
./create_english_sent_graphs.sh
./create_euca_sent_graphs.sh
./create_norec_sent_graphs
cd ..
```

Download the vectors

```
mkdir vectors
cd vectors
wget http://vectors.nlpl.eu/repository/20/58.zip
wget http://vectors.nlpl.eu/repository/20/32.zip
wget http://vectors.nlpl.eu/repository/20/34.zip
wget http://vectors.nlpl.eu/repository/20/18.zip
cd ..
```

Create mBERT representations for all datasets.
```
./do_bert.sh
```

The scripts folder contains the SLURM scripts to run all the experiments.

```
./scripts/run_SLURM_all_BERT.sh
./scripts/run_SLURM_no_BERT.sh
```


requirements
---
1. python3
2. pytorch
3. matplotlib
4. sklearn
5. gensim
6. numpy
7. h5py
