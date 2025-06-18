#!/bin/bash
#SBATCH --job-name=box_example
#SBATCH --output=box_example_%j.out
#SBATCH --error=box_example_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module load python
srun python3 /workspaces/RandomMedia/examples/box_example.py

module load python
#!/bin/bash
#SBATCH --job-name=box_example
#SBATCH --output=box_example_%j.out
#SBATCH --error=box_example_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

source /cm/shared/apps/anaconda3-2024.06-1/etc/profile.d/conda.sh
conda activate pmp

mpirun -n 32 python3 examples/box_example.py