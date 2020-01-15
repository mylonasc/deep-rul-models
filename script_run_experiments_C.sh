#!/bin/bash
etype=(A B C)
clossv=(0 0.02 0.05 0.09 0.1 0.15)

# GOALS: Figure out if 
#  *  I can quantify uncertainty, and have larger uncertainty for points out of the training sample
#  *  test some network configurations - configs:
#     A:    dilrates:  4, 8,16,32   (single densenet) - the one that seemed to work well in "B" experiments
#     B:    dilrates:  8, 8, 8, 8    (single densenet) 
#     C:    dilrates: 16,16,16,16 (single densenet)

for etypev in ${etype[@]}
do
  for closs in ${clossv[@]}
    do
	python3 run_experiment_C.py $closs $etypev> Experiments_C_closs_$closs-CType-$etypev.log
    done
done




