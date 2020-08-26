#!/bin/bash
#
#SBATCH --job-name=negscope --account=nn9447k
#SBATCH --output=negscope.out
#SBATCH --partition=accel --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem 8GB
#XSBATCH --mail-type=ALL
#SBATCH --mail-user=jeremycb@ifi.uio.no
#

module purge
source deactivate

#module load  Python/3.7.0-anaconda-5.3.0-extras-nsc1;
module use -a /cluster/shared/nlpl/software/modules/etc
module load nlpl-pytorch/1.5.0/3.7
module load nlpl-gensim/3.8.2/3.7

#source deactivate
#source activate py37;

CONFIG=$1;shift
D=$1;shift
O=$1;shift

DIR=experiments/$D
pwd
rm -rf $DIR
mkdir $DIR

echo $@

#python ./src/main.py --config $CONFIG --dir $DIR $@

python ./src/main.py --config configs/base.cfg --train data/sherlock_2/sp06/cdt.conllup --val data/sherlock_2/sp06/cdd.conllup --predict_file data/sherlock_2/sp06/cde.conllup --dir $DIR


python test.py conllup-starsem $DIR/cdd.conllup.pred > $DIR/cdd.starsem.pred
python test.py conllup-starsem $DIR/cde.conllup.pred > $DIR/cde.starsem.pred

perl scripts/eval.cd-sco.pl -g data/sherlock_2/sp06/cde.starsem.new -s $DIR/cde.starsem.pred > $DIR/cde.scores
perl scripts/eval.cd-sco.pl -g data/sherlock_2/sp06/cdd.starsem.new -s $DIR/cdd.starsem.pred > $DIR/cdd.scores

rm -f $DIR/*.save
rm -f $DIR/*.pk
# Script ends here
