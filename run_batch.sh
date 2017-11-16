#!/bin/bash

# options with defaults:
# --lstm_size 64
# --dropout 0.9
# --epochs 5
# --padding 100
# --batch_size=32
# --pre_seed 12345
# --run 1
# --runs 3
# --embedding w2v
# --optimizer adam | SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, TFOptimizer
# --loss binary_crossentropy | mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error,
# 							 | squared_hinge, hinge, categorical_hinge, logcosh, categorical_crossentropy, sparse_categorical_crossentropy,
#							 | binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity
# --activation1 relu		 | softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, sigmoid, linear
# --activation2 sigmoid		 | * AdvancedActivationLayers: LeakyReLU, PReLU, ELU, ThresholdedReLU
# --vsplit 0.1
# --rich 3 | 0, 1, 2, 3

#OMP_NUM_THREADS=2
#THEANO_FLAGS=device=cuda0
#floatX=float32
#device=gpu
ev="THEANO_FLAGS='device=cpu,warn.round=False' KERAS_BACKEND=theano"
#ev="KERAS_BACKEND=tensorflow"
cmd="env-python3-keras110/bin/python3 main.py"

# args[0]=" --rich 0"
# args[1]=" --rich 1"
# args[2]=" --rich 2"
# args[3]=" --rich 3"

args[0]="--epochs 8 batch_size 16 --optimizer RMSprop"
args[1]="--epochs 8 batch_size 16 --vsplit 0.2"
args[2]="--epochs 8 batch_size 16 --optimizer RMSprop --vsplit 0.2"
args[3]="--epochs 8 batch_size 16"


# test default
# args[0]="--epochs 2 --runs 3 --run 1 --batch_size 64 --padding 50"

eval ${ev} ${cmd} ${args[0]} &
eval ${ev} ${cmd} ${args[1]} &
eval ${ev} ${cmd} ${args[2]} &
eval ${ev} ${cmd} ${args[3]} &
#wait
#eval ${ev} ${cmd} ${args[4]} &
#eval ${ev} ${cmd} ${args[5]} &
#eval ${ev} ${cmd} ${args[6]} &
#eval ${ev} ${cmd} ${args[7]} &
#wait
#eval ${ev} ${cmd} ${args[8]} &
#eval ${ev} ${cmd} ${args[9]} &
#eval ${ev} ${cmd} ${args[10]} &
#eval ${ev} ${cmd} ${args[11]} &

# THEANO_FLAGS='device=cpu,warn.round=False' KERAS_BACKEND=theano env-python3-keras110/bin/python3 main.py --epochs 1 --runs 1 --run 1 --batch_size 64 --padding 50
# cd workspace/sem_train/experiments/src/main/python/
# source env-python3-keras110/bin/activate
# bash ./run_batch.sh