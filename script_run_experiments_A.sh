#!/bin/bash
dilratesa=(5 10)
ksizea=(5 15)
nfilta=(20 10)
dilratesb=(2 4)
ksizeb=(5 10)
nfiltb=(10)

for dra in ${dilratesa[@]}
do
	for ksa in ${ksizea[@]}
	do
		for nfa in ${nfilta[@]}
		do
			for drb in ${dilratesb[@]}
			do
				for ksb in ${ksizeb[@]}
				do
					for nfb in ${nfiltb[@]}
					do
						echo "Writing log at:"
					       	echo Experiment_$dra\_$ksa\_$nfa\_$drb\_$ksb\_$nfb.log
						python3 run_experiment_A.py $dra $ksa $nfa $drb $ksb $nfb > Experiment\_$dra\_$ksa\_$nfa\_$drb\_$ksb\_$nfb.log
					done
				done
			done
		done
	done
done





