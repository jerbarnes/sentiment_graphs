for r in 1 2 3 4 5 6 7 8 9 0;
do

## only pred
#for i in dt Turku18 sp06;
##for i in sp06;
#do
##for j in dpa+; do
#echo $r $i `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.predcue_only.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --target_style scope ;
#done

# only gold
for i in dt Turku18 sp06;
#for i in sp06;
do
for j in simple dpa dpa+ gcn; do
#for j in dpa+; do
echo $r $i $j `ls configs/base.cfg`
bash scripts/run_SLURM.sh \
    configs/base.cfg $i.goldcue_${j}_only.$r $i \
    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
    --use_elmo True \
    --help_style cue --target_style scope- \
    --bridge $j;
done
done

## syn pred
#for i in dt Turku18 sp06;
##for i in sp06;
#do
#for j in simple dpa dpa+ gcn; do
##for j in dpa+; do
#echo $r $i $j `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.predcue_${j}_syn.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --help_style syn --target_style scope \
#    --bridge $j;
#done
#done
#
## syn gold
#for i in dt Turku18 sp06;
##for i in sp06;
#do
#for j in simple dpa dpa+ gcn; do
##for j in dpa+; do
#echo $r $i $j `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.goldcue_${j}_syn.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --help_style syn,cue --target_style scope- \
#    --bridge $j;
#done
#done
#
## sem pred
#for i in dm_dt;
##for i in sp06;
#do
#for j in simple dpa dpa+ gcn; do
##for j in dpa+; do
#echo $r $i $j `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.predcue_${j}_sem.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --help_style sem --target_style scope \
#    --bridge $j;
#done
#done
#
## syn gold
#for i in dm_dt;
##for i in sp06;
#do
#for j in simple dpa dpa+ gcn; do
##for j in dpa+; do
#echo $r $i $j `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.goldcue_${j}_sem.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --help_style sem,cue --target_style scope- \
#    --bridge $j;
#done
#done
#
## sem+syn pred
#for i in dm_dt;
##for i in sp06;
#do
#for j in simple dpa dpa+ gcn; do
##for j in dpa+; do
#echo $r $i $j `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.predcue_${j}_sem+syn.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --help_style sem,syn --target_style scope \
#    --bridge $j;
#done
#done
#
## syn gold
#for i in dm_dt;
##for i in sp06;
#do
#for j in simple dpa dpa+ gcn; do
##for j in dpa+; do
#echo $r $i $j `ls configs/base.cfg`
#bash scripts/run_SLURM.sh \
#    configs/base.cfg $i.goldcue_${j}_sem+syn.$r $i \
#    --train ../egglayingwoolmilkpig/data/sherlock_2/$i/cdt.conllup \
#    --val ../egglayingwoolmilkpig/data/sherlock_2/$i/cdd.conllup \
#    --predict_file ../egglayingwoolmilkpig/data/sherlock_2/$i/cde.conllup \
#    --elmo_train ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdt.hdf5 \
#    --elmo_dev ../egglayingwoolmilkpig/data/elmo_embeds/$i/cdd.hdf5 \
#    --elmo_test ../egglayingwoolmilkpig/data/elmo_embeds/$i/cde.hdf5 \
#    --use_elmo True \
#    --help_style sem,syn,cue --target_style scope- \
#    --bridge $j;
#done
#done

done
