#!/bin/bash

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"

cd $ROOT

python  -u -m examples.crnn.train \
          --workers 32 --batch-size 100 --lr 1e-3 --warm-up 1000 --print-freq 100 --tfboard ./log/crnn \
          --train-root /data/plate/train/train_s.json \
          --val-root /data/plate/test/test_s.json \
          --checkpoint-dir ./checkpoints/crnn \
          --gpu 5   \
          --tfboard tensorboard \
          --model single |tee log1.txt 
