#!/bin/bash

# perform training in different batch sizes
python3 main.py \
    --batch_size 16 \
    --epochs 50 \
    --optimizer sgd \
    --learning_rate 0.01 \
    --momentum 0 \
    --weight_decay 0 \
    --loss_function softmax_and_cce \
    --preprocessing standardization \
    --model default

# remove any unecessary temporary files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|instance)" | xargs rm -rf