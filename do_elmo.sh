#!/bin/bash

#SBATCH --job-name=do_elmos
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=10GB

. ${HOME}/.bashrc;

if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export TMP=${SCRATCH}/tmp;
  mkdir -p ${TMP} > /dev/null 2>&1;
  export TMPDIR=${TMP};
fi


module purge
source deactivate

module load  Python/3.7.0-anaconda-5.3.0-extras-nsc1;
source deactivate
source activate py37;

{
m=../nlpl-vectors/english-elmo/;
outdir=../egglayingwoolmilkpig/data/elmo_embeds/;
for i in ../egglayingwoolmilkpig/data/sherlock_2/*/;
do 
    for j in t d e;
    do
        l=`ls $i/cd$j.conllup`;
        k=${l%.*};
        x=${i%*/};
        x=${x##*/};
        echo python elmo_embed.py --model $m --indata $l --outdata ${outdir}${x}/cd$j.hdf5;
    done;
done
} | xargs -d \\n -n 1 -P 6 -t sh -c

