#!/bin/bash

DATASET=$1;
SETUP=$2;
RUN=$3;
SEED=$4;

# EXTERNAL EMBEDDINGS
############################
echo "###########################"
echo EXTERNAL EMBEDDINGS
echo "###########################"

if [ $DATASET == norec ]; then
    EXTERNAL=vectors/20/58.zip
elif [ $DATASET == eu ]; then
    EXTERNAL=vectors/20/32.zip
elif [ $DATASET == ca ]; then
    EXTERNAL=vectors/20/34.zip
elif [ $DATASET == mpqa ]; then
    EXTERNAL=vectors/20/18.zip
elif [ $DATASET == ds_unis ]; then
    EXTERNAL=vectors/20/18.zip
else
    echo "NO EMBEDDINGS SUPPLIED FOR THIS DATASET"
    echo "EXITING TRAINING PROCEDURE"
    exit
fi

echo using external vectors: $EXTERNAL
echo

# INPUT FILES
############################
echo "###########################"
echo INPUT FILES
echo "###########################"

TRAIN=data/sent_graphs/$DATASET/$SETUP/train.conllu
DEV=data/sent_graphs/$DATASET/$SETUP/dev.conllu
TEST=data/sent_graphs/$DATASET/$SETUP/test.conllu

echo train data: $TRAIN
echo dev data: $DEV
echo test data: $TEST
echo

# Contextualized Vectors
############################
echo "###########################"
echo Contextualized Vectors
echo "###########################"

TRAIN_CV=data/sent_graphs/$DATASET/train_bert.hdf5
DEV_CV=data/sent_graphs/$DATASET/dev_bert.hdf5
TEST_CV=data/sent_graphs/$DATASET/test_bert.hdf5

echo train CV: $TRAIN_CV
echo dev CV: $DEV_CV
echo test CV: $TEST_CV
echo

# OUTPUT DIR
############################
echo "###########################"
echo OUTPUT DIR
echo "###########################"

DIR=experiments/$DATASET/BERT/$SETUP/$RUN
echo saving experiment to $DIR

pwd
rm -rf $DIR
mkdir $DIR

python ./src/main.py --config configs/sgraph_bert.cfg --train $TRAIN --val $DEV --predict_file $TEST --dir $DIR --external $EXTERNAL --elmo_train $TRAIN_CV --elmo_dev $DEV_CV --elmo_test $TEST_CV --use_elmo True --seed $SEED --vec_dim 768

# The models can be quite big and eat up a lot of space
# so we delete them. If you want to keep the models, remove the next lines
rm $DIR/best_model.save
rm $DIR/last_epoch.save
