#!/bin/bash

# options with defaults:
# lstm_size=64, dropout=0.9, epochs=5, padding=100, 
# batch_size=32, pre_seed=12345, run=0, runs=3, embedding='w2v'
# run > 0 overrides runs

#nohup \
THEANO_FLAGS='warn.round=False' KERAS_BACKEND=theano \
env-python3-keras110/bin/python3 main.py \
--epochs 1 --runs 3 --batch_size 128 --lstm_size 32 --padding 50 \
&

#nohup \
THEANO_FLAGS='warn.round=False' KERAS_BACKEND=theano \
env-python3-keras110/bin/python3 main.py \
--epochs 1 --run 1 --batch_size 128 --lstm_size 32 --padding 50 \
&

#nohup \
THEANO_FLAGS='warn.round=False' KERAS_BACKEND=theano \
env-python3-keras110/bin/python3 main.py \
--epochs 1 --run 2 --batch_size 128 --lstm_size 32 --padding 50 \
&

#nohup \
THEANO_FLAGS='warn.round=False' KERAS_BACKEND=theano \
env-python3-keras110/bin/python3 main.py \
--epochs 1 --run 3 --batch_size 128 --lstm_size 32 --padding 50 \
& # > out4.log &
