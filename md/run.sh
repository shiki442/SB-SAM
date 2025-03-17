#!/bin/bash

#SBATCH -A yangzhijian
#SBATCH -J MD-900K
#SBATCH --partition=pub
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

# Complie the program
make clean_exe
make
make clean

# Create output directory
output_dir="../../SAM_dataset/data12/900K"
mkdir -p $output_dir

# Run the program with the parameter file
./fe211 900.0 $output_dir

echo End at `date`