#!/bin/bash
#PBS -l select=1:ncpus=4:mem=4gb:arch=ivybridge
#PBS -l walltime=04:59:00
#PBS -N semeval
#PBS -A semeval18-12
#PBS -m abe
#PBS -M andreas.funke@uni-duesseldorf.de
#PBS -r y

me=`basename $0`
LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID"_"$PBS_ARRAY_INDEX".log"

SCRATCHDIR=/scratch_gs/$USER/semeval/out/ #$PBS_JOBID/
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
module load Theano/0.8.2
module load TensorFlow/1.4.0
module load Keras/2.1.2

shopt -s extglob
job[0]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 8 --out_path $SCRATCHDIR"
job[1]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 16 --out_path $SCRATCHDIR"
job[2]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 32 --out_path $SCRATCHDIR"
job[3]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 64 --out_path $SCRATCHDIR"
job[4]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 96 --out_path $SCRATCHDIR"
job[5]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 128 --out_path $SCRATCHDIR"
job[6]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 188 --out_path $SCRATCHDIR"
job[7]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 256 --out_path $SCRATCHDIR"
job[8]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 384 --out_path $SCRATCHDIR"
job[9]="KERAS_BACKEND=tensorflow python main.py --runs 5 --lstm_size 512 --out_path $SCRATCHDIR"
job[10]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.0 --out_path $SCRATCHDIR"
job[11]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.1 --out_path $SCRATCHDIR"
job[12]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.2 --out_path $SCRATCHDIR"
job[13]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.3 --out_path $SCRATCHDIR"
job[14]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.4 --out_path $SCRATCHDIR"
job[15]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.5 --out_path $SCRATCHDIR"
job[16]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.6 --out_path $SCRATCHDIR"
job[17]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.7 --out_path $SCRATCHDIR"
job[18]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.8 --out_path $SCRATCHDIR"
job[19]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 0.9 --out_path $SCRATCHDIR"
job[20]="KERAS_BACKEND=tensorflow python main.py --runs 5 --dropout 1.0 --out_path $SCRATCHDIR"
job[21]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 10 --out_path $SCRATCHDIR"
job[22]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 25 --out_path $SCRATCHDIR"
job[23]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 50 --out_path $SCRATCHDIR"
job[24]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 75 --out_path $SCRATCHDIR"
job[25]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 100 --out_path $SCRATCHDIR"
job[26]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 125 --out_path $SCRATCHDIR"
job[27]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 150 --out_path $SCRATCHDIR"
job[28]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 175 --out_path $SCRATCHDIR"
job[29]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 200 --out_path $SCRATCHDIR"
job[30]="KERAS_BACKEND=tensorflow python main.py --runs 5 --padding 250 --out_path $SCRATCHDIR"
job[31]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 1 --out_path $SCRATCHDIR"
job[32]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 2 --out_path $SCRATCHDIR"
job[33]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 4 --out_path $SCRATCHDIR"
job[34]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 8 --out_path $SCRATCHDIR"
job[35]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 16 --out_path $SCRATCHDIR"
job[36]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 32 --out_path $SCRATCHDIR"
job[37]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 64 --out_path $SCRATCHDIR"
job[38]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 128 --out_path $SCRATCHDIR"
job[39]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 256 --out_path $SCRATCHDIR"
job[40]="KERAS_BACKEND=tensorflow python main.py --runs 5 --batch_size 512 --out_path $SCRATCHDIR"
job[41]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 1 --out_path $SCRATCHDIR"
job[42]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 2 --out_path $SCRATCHDIR"
job[43]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 3 --out_path $SCRATCHDIR"
job[44]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 4 --out_path $SCRATCHDIR"
job[45]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 5 --out_path $SCRATCHDIR"
job[46]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 6 --out_path $SCRATCHDIR"
job[47]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 7 --out_path $SCRATCHDIR"
job[48]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 8 --out_path $SCRATCHDIR"
job[49]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 9 --out_path $SCRATCHDIR"
job[50]="KERAS_BACKEND=tensorflow python main.py --runs 5 --pre_seed 10 --out_path $SCRATCHDIR"
job[51]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer sgd --out_path $SCRATCHDIR"
job[52]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer rmsprop --out_path $SCRATCHDIR"
job[53]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer adagrad --out_path $SCRATCHDIR"
job[54]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer adadelta --out_path $SCRATCHDIR"
job[55]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer adam --out_path $SCRATCHDIR"
job[56]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer adamax --out_path $SCRATCHDIR"
job[57]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer nadam --out_path $SCRATCHDIR"
job[58]="KERAS_BACKEND=tensorflow python main.py --runs 5 --optimizer tfoptimizer --out_path $SCRATCHDIR"
job[59]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding d2v --out_path $SCRATCHDIR"
job[60]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding d2v_pf --out_path $SCRATCHDIR"
job[61]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding d2v_pf_lc --out_path $SCRATCHDIR"
job[62]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding d2v_pf_lc2 --out_path $SCRATCHDIR"
job[63]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding ftx --out_path $SCRATCHDIR"
job[64]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding ftx_pf --out_path $SCRATCHDIR"
job[65]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding ftx_pf_lc --out_path $SCRATCHDIR"
job[66]="KERAS_BACKEND=tensorflow python main.py --runs 5 --embedding ftx_pf_lc2 --out_path $SCRATCHDIR"
job[67]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss squared_hinge --out_path $SCRATCHDIR"
job[68]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss hinge --out_path $SCRATCHDIR"
job[69]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss categorical_hinge --out_path $SCRATCHDIR"
job[70]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss logcosh --out_path $SCRATCHDIR"
job[71]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss categorical_crossentropy --out_path $SCRATCHDIR"
job[72]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss sparse_categorical_crossentropy --out_path $SCRATCHDIR"
job[73]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss kullback_leibler_divergence --out_path $SCRATCHDIR"
job[74]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss poisson, cosine_proximity --out_path $SCRATCHDIR"
job[75]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss binary_crossentropy --out_path $SCRATCHDIR"
job[76]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss mean_squared_error --out_path $SCRATCHDIR"
job[77]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss mean_absolute_error --out_path $SCRATCHDIR"
job[78]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss mean_absolute_percentage_error --out_path $SCRATCHDIR"
job[79]="KERAS_BACKEND=tensorflow python main.py --runs 5 --loss mean_squared_logarithmic_error --out_path $SCRATCHDIR"
job[80]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 relu --out_path $SCRATCHDIR"
job[81]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 softmax --out_path $SCRATCHDIR"
job[82]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 elu --out_path $SCRATCHDIR"
job[83]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 selu --out_path $SCRATCHDIR"
job[84]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 softplus --out_path $SCRATCHDIR"
job[85]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 softsign --out_path $SCRATCHDIR"
job[86]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 tanh --out_path $SCRATCHDIR"
job[87]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 sigmoid --out_path $SCRATCHDIR"
job[88]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 linear --out_path $SCRATCHDIR"
job[89]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 leakyrelu --out_path $SCRATCHDIR"
job[90]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 prelu --out_path $SCRATCHDIR"
job[91]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 elu --out_path $SCRATCHDIR"
job[92]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation1 thresholdedrelu --out_path $SCRATCHDIR"
job[93]="KERAS_BACKEND=tensorflow python main.py --runs 5 --activation2 softmax --out_path $SCRATCHDIR"
job[94]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.05 --out_path $SCRATCHDIR"
job[95]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.1 --out_path $SCRATCHDIR"
job[96]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.15 --out_path $SCRATCHDIR"
job[97]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.2 --out_path $SCRATCHDIR"
job[98]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.25 --out_path $SCRATCHDIR"
job[99]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.3 --out_path $SCRATCHDIR"
job[100]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.35 --out_path $SCRATCHDIR"
job[101]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.4 --out_path $SCRATCHDIR"
job[102]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.45 --out_path $SCRATCHDIR"
job[103]="KERAS_BACKEND=tensorflow python main.py --runs 5 --vsplit 0.5 --out_path $SCRATCHDIR"
job[104]="KERAS_BACKEND=tensorflow python main.py --runs 5 --rich 0 --out_path $SCRATCHDIR"
job[105]="KERAS_BACKEND=tensorflow python main.py --runs 5 --rich 1 --out_path $SCRATCHDIR"
job[106]="KERAS_BACKEND=tensorflow python main.py --runs 5 --rich 2 --out_path $SCRATCHDIR"
job[107]=" --out_path $SCRATCHDIR"
job[108]=" --out_path $SCRATCHDIR"
job[109]=" --out_path $SCRATCHDIR"

# Select current call
run=${job[$PBS_ARRAY_INDEX]}
 
#cp -r $PBS_O_WORKDIR/* $SCRATCHDIR/.
#cd $SCRATCHDIR
 
echo >> $LOGFILE
echo "STARTING..." >> $LOGFILE
echo "---------------------------" >> $LOGFILE
 
cd semeval
eval $run

#cp -r "$SCRATCHDIR"out/* $PBS_O_WORKDIR/out/
cd $PBS_O_WORKDIR
 
echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE
 
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE
