#!/bin/bash

################ change forecasting horizon ################################ 
for FORECASTING_HORIZON in 20 60 120 240
do
    for MODEL in DenseGCNGRU GCNGRU GRU
    do
        for DATASET in GCS SEQ STADIUM_2023
        do
            # Call the experiment script with the specified parameters
            python main.py \
                --DATASET $DATASET \
                --MODEL $MODEL \
                --forecasting_horizon $FORECASTING_HORIZON \
                --save_model True \
                --save_dir './checkpoints'
        done
    done
done