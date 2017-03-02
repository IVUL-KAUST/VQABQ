#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch run        #
#                                  #
####################################

#SBATCH --account=k1216
#SBATCH --array=0
#SBATCH --workdir=/scratch/alfadlmm/VQA/
#SBATCH --partition=workq
#SBATCH --job-name=VQA    # Job name
#SBATCH --output=./log/%j.out # Stdout (%j expands to jobId)
#SBATCH --error=./log/%j.err # Stderr (%j expands to jobId)
#SBATCH --mem=128G   # memory per NODE
#SBATCH --time=1-00:00:00   # walltime

## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules
module load slurm

## RUN YOUR PROGRAM ##
srun ../python/Python-2.7.9/python generate.py
