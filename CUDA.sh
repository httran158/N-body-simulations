#!/bin/bash
#SBATCH --partition=gpu            # Quality of Service
#SBATCH --job-name=CUDA		   # Job Name
#SBATCH --time=00:10:00       	   # WallTime
#SBATCH --nodes=1      		   # Number of Nodes
#SBATCH --ntasks-per-node=1        # Number of tasks (MPI presseces)
#SBATCH --cpus-per-task=48         # Number of processors per task OpenMP threads()
#SBATCH --account=loni_cosc62004
module load cuda/11.0.2/intel-19.0.5
nvcc -o CUDA CUDA.cu -lcublas
./CUDA
