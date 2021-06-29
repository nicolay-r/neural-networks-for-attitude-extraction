#!/bin/bash
cd .. && CUDA_VISIBLE_DEVICES=0 python run_training.py --do-eval  \
    --bags-per-minibatch 32 --dropout-keep-prob 0.80 --cv-count 3  \
    --labels-count 3 --experiment rsr+ra --model-input-type ctx --ra-ver dbg \
    --model-name cnn --test-every-k-epoch 5 --learning-rate 0.1  \
    --balanced-input True --train-acc-limit 0.99  --epochs 100
