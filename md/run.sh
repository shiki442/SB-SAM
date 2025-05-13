#!/bin/bash

#SBATCH -A yangzhijian
#SBATCH -J MD-300K
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

# Run the program with the parameter file
# while IFS= read -r line; do
#     read -a ARGS <<< "$line"
#     mkdir -p "${ARGS[0]}"
#     ./fe211 "${ARGS[@]}"
# done < args.txt

# Read the specified line from args.txt
LINE_NUM=1
line=$(sed -n "${LINE_NUM}p" args.txt)
read -a ARGS <<< $line
mkdir -p "${ARGS[0]}"
./fe211 ${ARGS[@]}

echo End at `date`