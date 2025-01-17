#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH --gres=gpu:1
#SBATCH -J SAM
#SBATCH -o ./log/job-%j.out
#SBATCH -e ./log/job-%j.err
#SBATCH -p gpu

RUN_PATH="."
cd "$RUN_PATH" || exit 1

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo ---------------------------------------------
echo Time is `date`

/project/songpengcheng/miniconda3/envs/torch/bin/python -u main.py --config=params4.yml
echo End at `date`

