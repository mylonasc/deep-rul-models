# experiment_metadata = {"hpr_id" : "1", "description" : "Figure out if gated_tanh is beneficial. See effect of pre-network width on performance (100 50 30). See  effect of GN state on performance (30 15). No investigation on training schedule (training schedule fixed)."}

#ACT=(gated_tanh tanh relu)
ACT=(relu)

GNSTATE=(30 15)
NW_WIDTH=(30 50 100)

for ACT_ in ${ACT[@]}
  do
  for GNSTATE_ in ${GNSTATE[@]}
  do
    for NW_WIDTH_ in ${NW_WIDTH[@]}
    do
      echo $GNSTATE_ $NW_WIDTH_ $ACT_
      python3 run_experiment_graphnet.py $GNSTATE_ $NW_WIDTH_ $ACT_
    done
  done
done
  		



