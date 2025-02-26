#!/bin/bash

#SBATCH --partition=hpxg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

module load intel/parallelstudio/2019

cd $SLURM_SUBMIT_DIR

make clean

make

./fe211