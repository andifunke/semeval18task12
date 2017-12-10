#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb:arch=ivybridge
#PBS -l walltime=04:59:00
#PBS -N semeval
#PBS -A semeval18-12
#PBS -m abe
#PBS -M andreas.funke@uni-duesseldorf.de
#PBS -r y

me=`basename $0`
LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID"_"$PBS_ARRAY_INDEX".log"
 
SCRATCHDIR=/scratch_gs/$USER/semeval/ #$PBS_JOBID/
mkdir -p "$SCRATCHDIR"

cd $PBS_O_WORKDIR
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" START" > $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE
 
echo >> $LOGFILE
echo "GLOBAL PARAMETERS" >> $LOGFILE
echo "---------------------------" >> $LOGFILE
echo "Node      : "`hostname` >> $LOGFILE
echo "RunDir    : "$PBS_O_WORKDIR >> $LOGFILE
echo "ScratchDir: "$SCRATCHDIR >> $LOGFILE
echo "# CPUs    : "$NCPUS >> $LOGFILE
echo "# Threads : "$OMP_NUM_THREADS >> $LOGFILE

#cp -r $PBS_O_WORKDIR/semeval/* $SCRATCHDIR/.
#cd $SCRATCHDIR
#rm $PBS_JOBNAME"."$PBS_JOBID".log"

## Software-Umgebung laden
module load Python/3.4.5
python --version
module load Theano/0.8.2
module load Keras/2.1.2

cd semeval
shopt -s extglob
job[0]=''
job[1]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 1 --save_path $SCRATCHDIR'
job[2]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 2 --save_path $SCRATCHDIR'
job[3]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 3 --save_path $SCRATCHDIR'
job[4]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 4 --save_path $SCRATCHDIR'
job[5]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 5 --save_path $SCRATCHDIR'
job[6]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 6 --save_path $SCRATCHDIR'
job[7]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 7 --save_path $SCRATCHDIR'
job[8]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 8 --save_path $SCRATCHDIR'
job[9]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 9 --save_path $SCRATCHDIR'
job[10]='KERAS_BACKEND=theano python main.py --runs 5 --pre_seed 10 --save_path $SCRATCHDIR'
job[11]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer sgd --save_path $SCRATCHDIR'
job[12]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer rmsprop --save_path $SCRATCHDIR'
job[13]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer adagrad --save_path $SCRATCHDIR'
job[14]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer adadelta --save_path $SCRATCHDIR'
job[15]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer adam --save_path $SCRATCHDIR'
job[16]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer adamax --save_path $SCRATCHDIR'
job[17]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer nadam --save_path $SCRATCHDIR'
job[18]='KERAS_BACKEND=theano python main.py --runs 5 --optimizer tfoptimizer --save_path $SCRATCHDIR'
job[19]='KERAS_BACKEND=theano python main.py --runs 5 --embedding d2v --save_path $SCRATCHDIR'
job[20]='KERAS_BACKEND=theano python main.py --runs 5 --embedding d2v_pf --save_path $SCRATCHDIR'
job[21]='KERAS_BACKEND=theano python main.py --runs 5 --embedding d2v_pf_lc --save_path $SCRATCHDIR'
job[22]='KERAS_BACKEND=theano python main.py --runs 5 --embedding d2v_pf_lc2 --save_path $SCRATCHDIR'
job[23]='KERAS_BACKEND=theano python main.py --runs 5 --embedding ftx --save_path $SCRATCHDIR'
job[24]='KERAS_BACKEND=theano python main.py --runs 5 --embedding ftx_pf --save_path $SCRATCHDIR'
job[25]='KERAS_BACKEND=theano python main.py --runs 5 --embedding ftx_pf_lc --save_path $SCRATCHDIR'
job[26]='KERAS_BACKEND=theano python main.py --runs 5 --embedding ftx_pf_lc2 --save_path $SCRATCHDIR'
job[27]='KERAS_BACKEND=theano python main.py --runs 5 --loss squared_hinge --save_path $SCRATCHDIR'
job[28]='KERAS_BACKEND=theano python main.py --runs 5 --loss hinge --save_path $SCRATCHDIR'
job[29]='KERAS_BACKEND=theano python main.py --runs 5 --loss categorical_hinge --save_path $SCRATCHDIR'
job[30]='KERAS_BACKEND=theano python main.py --runs 5 --loss logcosh --save_path $SCRATCHDIR'
job[31]='KERAS_BACKEND=theano python main.py --runs 5 --loss categorical_crossentropy --save_path $SCRATCHDIR'
job[32]='KERAS_BACKEND=theano python main.py --runs 5 --loss sparse_categorical_crossentropy --save_path $SCRATCHDIR'
job[33]='KERAS_BACKEND=theano python main.py --runs 5 --loss kullback_leibler_divergence --save_path $SCRATCHDIR'
job[34]='KERAS_BACKEND=theano python main.py --runs 5 --loss poisson, cosine_proximity --save_path $SCRATCHDIR'
job[35]='KERAS_BACKEND=theano python main.py --runs 5 --loss binary_crossentropy --save_path $SCRATCHDIR'
job[36]='KERAS_BACKEND=theano python main.py --runs 5 --loss mean_squared_error --save_path $SCRATCHDIR'
job[37]='KERAS_BACKEND=theano python main.py --runs 5 --loss mean_absolute_error --save_path $SCRATCHDIR'
job[38]='KERAS_BACKEND=theano python main.py --runs 5 --loss mean_absolute_percentage_error --save_path $SCRATCHDIR'
job[39]='KERAS_BACKEND=theano python main.py --runs 5 --loss mean_squared_logarithmic_error --save_path $SCRATCHDIR'
job[40]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 relu --save_path $SCRATCHDIR'
job[41]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 softmax --save_path $SCRATCHDIR'
job[42]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 elu --save_path $SCRATCHDIR'
job[43]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 selu --save_path $SCRATCHDIR'
job[44]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 softplus --save_path $SCRATCHDIR'
job[45]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 softsign --save_path $SCRATCHDIR'
job[46]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 tanh --save_path $SCRATCHDIR'
job[47]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 sigmoid --save_path $SCRATCHDIR'
job[48]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 linear --save_path $SCRATCHDIR'
job[49]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 leakyrelu --save_path $SCRATCHDIR'
job[50]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 prelu --save_path $SCRATCHDIR'
job[51]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 elu --save_path $SCRATCHDIR'
job[52]='KERAS_BACKEND=theano python main.py --runs 5 --activation1 thresholdedrelu --save_path $SCRATCHDIR'
job[53]='KERAS_BACKEND=theano python main.py --runs 5 --activation2 softmax --save_path $SCRATCHDIR'
job[54]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.05 --save_path $SCRATCHDIR'
job[55]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.1 --save_path $SCRATCHDIR'
job[56]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.15 --save_path $SCRATCHDIR'
job[57]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.2 --save_path $SCRATCHDIR'
job[58]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.25 --save_path $SCRATCHDIR'
job[59]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.3 --save_path $SCRATCHDIR'
job[60]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.35 --save_path $SCRATCHDIR'
job[61]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.4 --save_path $SCRATCHDIR'
job[62]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.45 --save_path $SCRATCHDIR'
job[63]='KERAS_BACKEND=theano python main.py --runs 5 --vsplit 0.5 --save_path $SCRATCHDIR'
job[64]='KERAS_BACKEND=theano python main.py --runs 5 --rich 0 --save_path $SCRATCHDIR'
job[65]='KERAS_BACKEND=theano python main.py --runs 5 --rich 1 --save_path $SCRATCHDIR'
job[66]='KERAS_BACKEND=theano python main.py --runs 5 --rich 2 --save_path $SCRATCHDIR'
job[67]=''
job[68]=''
job[69]=''
 
# Select current call
run=${job[$PBS_ARRAY_INDEX]}
 
#cp -r $PBS_O_WORKDIR/* $SCRATCHDIR/.
#cd $SCRATCHDIR
 
echo >> $LOGFILE
echo "STARTING..." >> $LOGFILE
echo "---------------------------" >> $LOGFILE
 
eval $run

#cp -r "$SCRATCHDIR"out/* $PBS_O_WORKDIR/out/
cd $PBS_O_WORKDIR
 
echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE
 
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE
