#!/bin/bash

# options with defaults:
# lstm_size=64, dropout=0.9, epochs=5, padding=100, 
# batch_size=32, pre_seed=12345, run=1, runs=1, embedding='w2v'

ev="THEANO_FLAGS='warn.round=False' KERAS_BACKEND=theano"
cmd="env-python3-keras110/bin/python3 main.py"

args[0]="--epochs 8 --runs 3 --run 1 --dropout 0.5"
args[1]="--epochs 8 --runs 3 --run 1 --dropout 0.6"
args[2]="--epochs 8 --runs 3 --run 1 --dropout 0.7"
args[3]="--epochs 8 --runs 3 --run 1 --dropout 0.8"
args[4]="--epochs 8 --runs 3 --run 1 --dropout 0.9"
args[5]="--epochs 8 --runs 3 --run 1 --dropout 0.95"
args[6]="--epochs 8 --runs 3 --run 1 --batch_size 8"
args[7]="--epochs 8 --runs 3 --run 1 --batch_size 16"
args[8]="--epochs 8 --runs 3 --run 1 --batch_size 32"
args[9]="--epochs 8 --runs 3 --run 1 --padding 75"
args[10]="--epochs 8 --runs 3 --run 1 --padding 300"
args[11]="--epochs 8 --runs 3 --run 1 --padding 500"
args[12]="--epochs 10 --runs 3 --run 1 --padding 300"
args[13]="--epochs 10 --runs 3 --run 1 --batch_size 8"
args[14]="--epochs 10 --runs 3 --run 1 --padding 300 --batch_size 8"

eval ${ev} ${cmd} ${args[0]} &
eval ${ev} ${cmd} ${args[1]} &
eval ${ev} ${cmd} ${args[2]} &
wait
eval ${ev} ${cmd} ${args[3]} &
eval ${ev} ${cmd} ${args[4]} &
eval ${ev} ${cmd} ${args[5]} &
wait
eval ${ev} ${cmd} ${args[6]} &
eval ${ev} ${cmd} ${args[7]} &
eval ${ev} ${cmd} ${args[8]} &
wait
eval ${ev} ${cmd} ${args[9]} &
eval ${ev} ${cmd} ${args[10]} &
eval ${ev} ${cmd} ${args[11]} &
wait
eval ${ev} ${cmd} ${args[12]} &
eval ${ev} ${cmd} ${args[13]} &
eval ${ev} ${cmd} ${args[14]} &
