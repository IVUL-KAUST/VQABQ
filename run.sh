#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch run        #
#                                  #
####################################

#SBATCH --array=0-99
#SBATCH --workdir=/home/alfadlmm/VQA/
#SBATCH --job-name=VQA    # Job name
#SBATCH --output=./log/%j.out # Stdout (%j expands to jobId)
#SBATCH --error=./log/%j.err # Stderr (%j expands to jobId)
#SBATCH --mem=220G   # memory per NODE
#SBATCH --time=2-00:00:00   # walltime

## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules
module load slurm
module load anaconda

## RUN YOUR PROGRAM ##
for i in {0..15}
do
	srun python generate.py $i &
done

wait