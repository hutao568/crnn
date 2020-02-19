#!/bin/bash

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"

cd $ROOT

python -u -m examples.crnn.train \
          --workers 16 --batch-size 512 --lr 4e-4 --warm-up 1000 --print-freq 100 --tfboard ./log/crnn \
          --train-root /data/plate/train/train_s.json \
          --val-root /data/plate/train/train_s.json \
          --evaluate \
          --checkpoint checkpoints/crnn1/checkpoint_best.pth.tar \
          --model single  \
          --h 32\
          --gpu 6

# --dist-url 'tcp://127.0.0.1:32123' --multiprocessing-distributed --world-size 1 --rank 0 \
