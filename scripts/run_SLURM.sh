#!/usr/bin/bash


CONFIG=$1; shift
NAME=$1; shift
rm -f logs/$NAME.out

#sbatch  -n 1 -t 24:00:00 --mem 8GB -o logs/$NAME.out -e logs/$NAME.out -J $NAME.parse run_X.sh $CONFIG $NAME $args;
#bash  ./scripts/run_Y.sh $CONFIG $NAME;
#sbatch  -o logs/$NAME.out -e logs/$NAME.out -J $NAME.parse ./run_Z.sh $CONFIG $NAME $@;

sbatch -o logs/new1.out -e logs/new1.out -J base.parse ./scripts/run_Z.sh configs/base.cfg NEWBASE
#bash run_X.sh $CONFIG $NAME $args;
