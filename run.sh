#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch run        #
#                                  #
####################################

#SBATCH --array=0-999
#SBATCH --job-name=VQA    # Job name
#SBATCH --output=gen.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=gen.%j.err # Stderr (%j expands to jobId)
#SBATCH --workdir=/home/alfadlmm/VQA/
#SBATCH --mem=128G   # memory per NODE
#SBATCH --time=2-00:00:00   # walltime

export I_MPI_FABRICS=shm:dapl

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules
module load anaconda

## RUN YOUR PROGRAM ##
srun python generate.py
