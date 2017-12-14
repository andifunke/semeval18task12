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
 
# parameterFile="pbs_job_parameter.txt"

# Select current call
# not working
params=`sed -n "${PBS_ARRAY_INDEX}q;d" job_params`
run=( $params )
 
#cp -r $PBS_O_WORKDIR/* $SCRATCHDIR/.
#cd $SCRATCHDIR
 
echo >> $LOGFILE
echo "STARTING..." >> $LOGFILE
echo "---------------------------" >> $LOGFILE

cd semeval
eval $run
 
#cp -r "$SCRATCHDIR"/* $PBS_O_WORKDIR/.
cd $PBS_O_WORKDIR
 
echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE
 
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE
