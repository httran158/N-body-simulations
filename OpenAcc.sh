#!/bin/bash
#SBATCH --partition=gpu        # Quality of Service
#SBATCH --job-name=OpenAcc     # Job Name
#SBATCH --time=24:00:00        # WallTime
#SBATCH --nodes=1              # Number of Nodes
#SBATCH --ntasks-per-node=1    # Number of tasks (MPI presseces)
#SBATCH --cpus-per-task=48     # Number of processors per task OpenMP threads()
#SBATCH --account=loni_cosc62004
module load pgi/21.3
pwd
echo "DIR=" $SLURM_SUBMIT_DIR
echo "TASKS_PER_NODE=" $SLURM_TASKS_PER_NODE
echo "NNODES=" $SLURM_NNODES
echo "NTASKS" $SLURM_NTASKS
echo "JOB_CPUS_PER_NODE" $SLURM_JOB_CPUS_PER_NODE
echo $SLURM_NODELIST
pgc++ -O3 -o OpenAcc OpenAcc.cpp -mp -acc -ta=tesla:cc70,keepgpu -Minfo=accel
./OpenAcc
