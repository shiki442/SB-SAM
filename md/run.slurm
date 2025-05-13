#!/bin/bash

#SBATCH --account yangzhijian
#SBATCH --job-name MD-data
#SBATCH --partition=pub
#SBATCH --output job-%A_%a.out
#SBATCH --error job-%A_%a.err
#SBATCH --array=1-30%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo ---------------------------------------------
echo Time is `date`

module load intel/parallelstudio/2019

cd $SLURM_SUBMIT_DIR

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" args.txt)

read -a ARGS <<< "$line"
# Create output directory
mkdir -p "${ARGS[0]}"

# Run the program with the parameter file
./fe211 ${ARGS[@]}

echo End at `date`