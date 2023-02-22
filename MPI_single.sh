#!/bin/bash
#SBATCH --qos=normal            # Quality of Service
#SBATCH --job-name=MPI          # Job Name
#SBATCH --time=01:00:00         # WallTime
#SBATCH --nodes=4               # Number of Nodes
#SBATCH --ntasks-per-node=1     # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=1       # Number of processors per task OpenMP threads()
#SBATCH --gres=mic:0            # Number of Co-Processors
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ttran25@tulane.edu
# setup module that the code needs
module load intel-psxe/2015-update1
pwd
echo "DIR=" $SLURM_SUBMIT_DIR
echo "TASKS_PER_NODE=" $SLURM_TASKS_PER_NODE
echo "NNODES=" $SLURM_NNODES
echo "NTASKS" $SLURM_NTASKS
echo "JOB_CPUS_PER_NODE" $SLURM_JOB_CPUS_PER_NODE
echo $SLURM_NODELIST
mpirun ./main
