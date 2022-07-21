#!/bin/bash
#SBATCH -p short
#SBATCH --open-mode=truncate
#SBATCH --parsable
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hamza_zerhouni@hms.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source /n/groups/patel/Hamza/python_3.6.0/bin/activate

python /n/groups/patel/Hamza/scripts/segmentation.py $1 $2 && echo "PYTHON SCRIPT COMPLETED"
