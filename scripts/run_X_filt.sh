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

python main.py --config $CONFIG --dir $DIR $@


python robutils.py $DIR/cde_filt.conllup.pred > $DIR/cde_filt.starsem.pred
python robutils.py $DIR/cdd_filt.conllup.pred > $DIR/cdd_filt.starsem.pred

perl eval.cd-sco.pl -g data/sherlock/cdd_filt.starsem -s $DIR/cdd_filt.starsem.pred
perl eval.cd-sco.pl -g data/sherlock/cde_filt.starsem -s $DIR/cde_filt.starsem.pred

# Script ends here
