#!/bin/bash
#
#XSBATCH -n 1
#XSBATCH -t 24:00:00
#XSBATCH --mem 8GB
#XSBATCH --mail-type=ALL
#XSBATCH --mail-user=robin.kurtz@liu.se
#

module purge
source deactivate

module load  Python/3.7.0-anaconda-5.3.0-extras-nsc1;
source activate py37;

CONFIG=$1;shift
D=$1;shift

DIR=experiments/$D

rm -rf $DIR
mkdir $DIR

echo $@

DATA=$1;shift

python main.py --config $CONFIG --dir $DIR --train data/sherlock/cdt.$DATA.conllup --val data/sherlock/cdd.$DATA.conllup --predict_file data/sherlock/cde.$DATA.conllup $@


python robutils.py $DIR/cde.$DATA.conllup.pred > $DIR/cde.$DATA.starsem.pred
python robutils.py $DIR/cdd.$DATA.conllup.pred > $DIR/cdd.$DATA.starsem.pred

perl eval.cd-sco.pl -g data/sherlock/cdd.$DATA.starsem -s $DIR/cdd.$DATA.starsem.pred > $DIR/cdd.eval
perl eval.cd-sco.pl -g data/sherlock/cde.$DATA.starsem -s $DIR/cde.$DATA.starsem.pred > $DIR/cde.eval

# Script ends here
