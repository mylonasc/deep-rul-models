# experiment_metadata = {"hpr_id" : "1", "description" : "Figure out if gated_tanh is beneficial. See effect of pre-network width on performance (100 50 30). See  effect of GN state on performance (30 15). No investigation on training schedule (training schedule fixed)."}

#ACT=(gated_tanh tanh relu)
ACT=(gated_tanh)

GNSTATE=(30 15)
NW_WIDTH=(30 100)
CNN_CONV_BLOCKS=(3)
CNN_KERNEL=(3)
CNN_NFILT1=(10 20 40)
CNN_NFILT2=(10 20 40)
echo "."


for CNN_NFILT1_ in ${CNN_NFILT1[@]}
  do
  for CNN_NFILT2_ in ${CNN_NFILT2[@]}
    do
      for CNN_KERNEL_ in ${CNN_KERNEL[@]}
      do
        for CNN_CONV_BLOCKS_ in ${CNN_CONV_BLOCKS[@]}
        do
        for ACT_ in ${ACT[@]}
          do
          for GNSTATE_ in ${GNSTATE[@]}
          do
            for NW_WIDTH_ in ${NW_WIDTH[@]}
            do
              echo $GNSTATE_ $NW_WIDTH_ $ACT_
              python3 run_experiment_graphnet.py $GNSTATE_ $NW_WIDTH_ $ACT_ $CNN_CONV_BLOCKS_ $CNN_NFILT1_ $CNN_NFILT2_ $CNN_KERNEL_
            done
          done
        done
      done
    done
  done
done

