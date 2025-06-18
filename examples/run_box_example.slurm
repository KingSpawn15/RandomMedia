#!/bin/bash
#SBATCH --job-name=box_example
#SBATCH --output=box_example_%j.out
#SBATCH --error=box_example_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module load python  # Or: module load python/3.x mpi4py
srun python3 examples/box_example.py