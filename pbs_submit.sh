#!/bin/bash

## select = #of nodes; ncpus ? #of cores
#PBS -l select=1:ncpus=20:mem=30GB:arch=ivybridge
#PBS -l walltime=02:59:00
## restart if job fails
#PBS -r n
#PBS -N semeval
#PBS -A semeval18-12
#PBS -m abe
#PBS -M andreas.funke@uni-duesseldorf.de

## Log-File definieren
export LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID".log"
touch $LOGFILE

## Scratch-Laufwerk definieren und erzeugen
SCRATCHDIR=/scratch_gs/funkea/semeval/$PBS_JOBID
mkdir -p $SCRATCHDIR

## Information zum Start in das Log-File schreiben
cd $PBS_O_WORKDIR
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" START" > $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE 

NUMBER_NODES=$(wc -l < $PBS_NODEFILE)
echo "Requested $NUMBER_NODES nodes"
cat $PBS_NODEFILE
echo "Requested amount of CPUs (per node): $OMP_NUM_THREADS"

## Daten vom Arbeitsverzeichnis auf das Scratch-Laufwerk kopieren
cp -r $PBS_O_WORKDIR/semeval/* $SCRATCHDIR/.
cd $SCRATCHDIR
rm $PBS_JOBNAME"."$PBS_JOBID".log"

## Software-Umgebung laden
module load Python/3.4.5
module load Theano/1.0.1
module load Keras/2.1.2
python --version

## Python-Aufruf
#cd SCRATCHDIR
#KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 8 &
#KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 16 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 32 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 64 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 96 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 128 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 188 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 256 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 384 &
KERAS_BACKEND=theano python main.py --runs 5 --lstm_size 512 &
wait

## Daten zurück kopieren
cp -r "$SCRATCHDIR"/out/* $PBS_O_WORKDIR/out/.
cd $PBS_O_WORKDIR
 
## Verfügbare Informationen zum Auftrag in das Log-File schreiben
echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE  
 
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE
