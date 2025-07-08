#!/bin/bash
#SBATCH --job-name=box_example
#SBATCH --output=box_example_%j.out
#SBATCH --error=box_example_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu-long.q
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

source /cm/shared/apps/anaconda3-2024.06-1/etc/profile.d/conda.sh
conda activate pmp

# mpirun -n 32 python3 -u examples/free_space.py
mpirun -n 32 python3 -u examples/random_slab.py --run --modes 0
# mpirun -n 32 python3 -u examples/tmatrix_test.py --run