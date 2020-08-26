#!/bin/bash
#
#SBATCH --account=nn9447k
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --mem=16G

. ${HOME}/.bashrc;

module load PyTorch/1.0.1-fosscuda-2018b-Python-3.6.6;
source ~/mypy/bin/activate;

CONFIG=$1;shift
D=$1;shift

python ./src/main.py --config $CONFIG

exit 0
# Script ends here
