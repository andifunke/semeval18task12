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
 
#SCRATCHDIR=/scratch_gs/$USER/semeval/$PBS_JOBID/
#mkdir -p "$SCRATCHDIR"

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
module load Theano/1.0.1
module load Keras/2.1.2
python --version

shopt -s extglob
parameterFile="pbs_job_parameter.txt"
 
# Select current call
run=$(sed "${PBS_ARRAY_INDEX} p" $parameterFile)
 
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
