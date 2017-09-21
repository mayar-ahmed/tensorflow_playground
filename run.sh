#!/usr/bin/env bash

python classify_cifar_10.py \
    --model=Basic \
    --num_epochs=5 \
    --batch_size=16 \
    --learning_rate=0.0001 \
    --data_dir=cifar10 \
    --exp_dir=experiment_1 \
    --train_n_test=True \
    --is_train=True