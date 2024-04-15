#!/bin/bash

# Experiment 1
# perform against different batch sizes
# for i in $(seq 1 3);
# do
#     for batch_size in {1,2,4,8,16,32,64,128};
#     do
#         echo "start training with batch size $batch_size"

#         # perform training in different batch sizes
#         python3 main.py \
#             --batch_size $batch_size \
#             --epochs 50 \
#             --optimizer sgd \
#             --learning_rate 0.01 \
#             --momentum 0 \
#             --weight_decay 0 \
#             --loss_function softmax_and_cce \
#             --preprocessing standardization >> $i\_log$batch_size.txt
        
#         echo "finish training with $batch_size"
#         echo ""

#         # change the name of the saved model
#         mv model.zip $i\_$batch_size\_model.zip
#     done
# done

# Experiment 2
# perform against different optimizers
# adam
# for i in $(seq 1 3);
# do
#     echo "start training with Adam optimizer"

#     # perform training using Adam optimizer
#     python3 main.py \
#         --batch_size 1 \
#         --epochs 50 \
#         --optimizer adam \
#         --learning_rate 0.01 \
#         --loss_function softmax_and_cce \
#         --preprocessing standardization >> $i\_log\_adam.txt
    
#     echo "finish training with adam"
#     echo ""

#     # change the name of the saved model
#     mv model.zip $i\_adam_model.zip
# done
# # sgd (+ momentum)
# for i in $(seq 1 3);
# do
#     for momentum in {0,0.1,0.2,0.5,0.9};
#     do
#         echo "start training with SGD (+ Momentum) optimizer"

#         # perform training using SGD (+ Momentum) optimizer
#         python3 main.py \
#             --batch_size 1 \
#             --epochs 50 \
#             --optimizer sgd \
#             --learning_rate 0.01 \
#             --momentum $momentum \
#             --weight_decay 0 \
#             --loss_function softmax_and_cce \
#             --preprocessing standardization >> $i\_log\_sgd\_mom=$momentum.txt

#         echo "finish training with sgd $momentum"
#         echo ""

#         # change the name of the saved model
#         mv model.zip $i\_sgd_mom\_$momentum\_model.zip
#     done
# done

# Experiment 3 (only left with 1 more time)
# perform against different learning rates
for i in $(seq 3 3);
do
    for learning_rate in {0.1,0.05,0.01,0.005,0.001,0.0001};
    do
        echo "start training with learning rate $learning_rate"

        # perform training using different learning rates
        python3 main.py \
            --batch_size 1 \
            --epochs 50 \
            --optimizer adam \
            --learning_rate $learning_rate \
            --momentum 0 \
            --weight_decay 0 \
            --loss_function softmax_and_cce \
            --preprocessing standardization >> $i\_log\_adam\_lr=$learning_rate.txt

        echo "finish training with lr $learning_rate"
        echo ""

        # change the name of the saved model
        mv model.zip $i\_adam\_lr\_$learning_rate\_model.zip
    done
done

# Experiment 4
# perform against different weight decays
for i in $(seq 1 3);
do
    for weight_decay in {0.1,0.01,0.001,0.0001};
    do
        echo "start training with weight decay $weight_decay"

        # perform training using different weight decays
        python3 main.py \
            --batch_size 1 \
            --epochs 50 \
            --optimizer sgd \
            --learning_rate 0.01 \
            --momentum 0 \
            --weight_decay $weight_decay \
            --loss_function softmax_and_cce \
            --preprocessing standardization >> $i\_log\_sgd\_wd=$weight_decay.txt

        echo "finish training with weight decay $weight_decay"
        echo ""

        # change the name of the saved model
        mv model.zip $i\_sgd\_wd\_$weight_decay\_model.zip
    done
done

# Experiment 5
# ablation studies
# perform against different model structure (with Dropout, with BatchNorm, etc.)
# with Dropout
# with BatchNorm
# with different number of hidden layers
declare -a arr=("default" "single" "dropout" "batchnorm")
for i in $(seq 1 3);
do
    for model in "${arr[@]}";
    do
        echo "start training with model $model [$i]"

        # # perform training using different weight decays
        python3 main.py \
            --batch_size 1 \
            --epochs 50 \
            --optimizer adam \
            --learning_rate 0.01 \
            --momentum 0 \
            --weight_decay 0 \
            --loss_function softmax_and_cce \
            --preprocessing standardization \
            --model $model >> $i\_log\_adam\_model=$model.txt

        echo "finish training with model $model [$i]"
        echo ""

        # change the name of the saved model
        mv model.zip $i\_adam\_$model\_model.zip
    done
done

# Experiment 6
# compare different losses


# remove any unecessary temporary files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|instance)" | xargs rm -rf
