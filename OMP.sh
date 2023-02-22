#!/bin/bash
#SBATCH --partition=defq 	 # Partition
#SBATCH --qos=long 		 # Quality of Service
#SBATCH --job-name=OMP  	 # Job Name
#SBATCH --time=24:00:00 	 # WallTime
#SBATCH --nodes=1   		 # Number of Nodes
#SBATCH --ntasks-per-node=1  	 # Number of tasks (MPI presseces)
#SBATCH --cpus-per-task=20    	 # Number of processors per task OpenMP threads()
#SBATCH --gres=mic:0  		 # Number of Co-Processors
module load gcc/6.3.0
echo $SLURM_NODELIST
export OMP_NUM_THREADS=20
g++ -o3 serial_omp.cpp -fopenmp
./a.out
