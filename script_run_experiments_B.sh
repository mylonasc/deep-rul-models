#!/bin/bash
clossv=(0.1 0.5 0.9)
clossv=(0 0.05)

for closs in ${clossv[@]}
do
	python3 run_experiment_B.py $closs > Experiments_B_closs_$closs.log
done




