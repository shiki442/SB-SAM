#!/bin/bash

#SBATCH -A yangzhijian
#SBATCH -J MD_make
#SBATCH --partition=hpib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo ---------------------------------------------
echo Time is `date`

module load intel/parallelstudio/2019

cd $SLURM_SUBMIT_DIR

make -f Makefile_n12 clean_exe

make -f Makefile_n12

make -f Makefile_n12 clean

echo End at `date`