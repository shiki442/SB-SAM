#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH --gres=gpu:4
#SBATCH -J SAM
#SBATCH -o ./log/job-%j.out
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=2

RUN_PATH="."
cd "$RUN_PATH" || exit 1

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo ---------------------------------------------
echo configuration file: params3d
echo Time is `date`

/project/songpengcheng/miniconda3/envs/torch/bin/python -u eval.py
echo End at `date`