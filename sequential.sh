#!/bin/bash
#SBATCH --job-name=sequential ### Job Name
#SBATCH --qos=normal          ### Quality of Service (like a queue in PBS)
#SBATCH --time=0-24:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=20  ### Number of tasks to be launched per Node
#SBATCH --cpus-per-task=1     ### Number of cores per task

module load intel-psxe/2016
mpirun ./sequential

module load anaconda3/2019.03
export MPLBACKEND=Agg
python ./Animation.py 50 $SLURM_NTASKS
