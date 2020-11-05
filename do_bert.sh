#!/bin/bash


# BERT models
#berts=( "bert-base-cased" "bert-base-multilingual-cased" "bert-base-multilingual-cased" "bert-base-multilingual-cased" );
berts=( "bert-base-multilingual-cased" );

# datasets
datadir="data/sent_graphs"
#datasets=( "ds_unis/head_final" "norec/head_final-inside_label" "ca/head_final" "eu/head_final" );
datasets=( "norec/head_final-inside_label" );


for ((i=0;i<${#berts[@]};++i)); do
  model="${berts[i]}"
  for t in train dev test; do
    indata="$datadir"/"${datasets[i]}"/"$t".conllu
    outfile="$datadir"/"${datasets[i]}"/"$t"_bert.hdf5
    printf "Using %s for %s\n" "$model" "$indata";
    printf "Saving to %s\n" "$outfile"
    python3 bert_embed.py --model $model --indata $indata --outdata $outfile;
  done;
  echo
done;




