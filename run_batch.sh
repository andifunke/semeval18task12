#!/bin/bash

# options with defaults:
# --lstm_size 64
# --dropout 0.9
# --epochs 5
# --padding 100
# --batch_size=32
# --pre_seed 12345
# --run 1
# --runs 1
# --embedding 'w2v'

#OMP_NUM_THREADS=2
#THEANO_FLAGS=device=cuda0
#floatX=float32
#device=gpu
# ev="OMP_NUM_THREADS=2 THEANO_FLAGS='warn.round=False' KERAS_BACKEND=theano"
ev="KERAS_BACKEND=tensorflow"
cmd="env-python3-keras110/bin/python3 main.py"

#args[0]="--epochs 8 --runs 3 --run 1 --dropout 0.5"
#args[1]="--epochs 8 --runs 3 --run 1 --dropout 0.6"
#args[2]="--epochs 8 --runs 3 --run 1 --dropout 0.7"
#args[3]="--epochs 8 --runs 3 --run 1 --dropout 0.8"
#args[4]="--epochs 8 --runs 3 --run 1 --dropout 0.9"
#args[5]="--epochs 8 --runs 3 --run 1 --dropout 0.95"
#args[6]="--epochs 8 --runs 3 --run 1 --batch_size 8"
#args[7]="--epochs 8 --runs 3 --run 1 --batch_size 16"
#args[8]="--epochs 8 --runs 3 --run 1 --batch_size 32"
#args[9]="--epochs 8 --runs 3 --run 1 --padding 75"
#args[10]="--epochs 8 --runs 3 --run 1 --padding 300"
#args[11]="--epochs 8 --runs 3 --run 1 --padding 500"
#args[12]="--epochs 10 --runs 3 --run 1 --padding 300"
#args[13]="--epochs 10 --runs 3 --run 1 --batch_size 8"
#args[14]="--epochs 10 --runs 3 --run 1 --padding 300 --batch_size 8"
args[0]="--epochs 2 --runs 3 --run 1 --batch_size 64 --padding 50"

eval ${ev} ${cmd} ${args[0]} &
