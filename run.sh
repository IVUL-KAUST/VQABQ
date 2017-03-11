#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch run        #
#                                  #
####################################

#SBATCH --account=k1216
#SBATCH --array=0-798
#SBATCH --partition=workq
#SBATCH --workdir=/scratch/alfadlmm/
#SBATCH --job-name=VQA    # Job name
#SBATCH --output=./log/%j.out # Stdout (%j expands to jobId)
#SBATCH --error=./log/%j.err # Stderr (%j expands to jobId)
#SBATCH --mem=120G   # memory per NODE
#SBATCH --time=09:00:00   # walltime

## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules
module load slurm

# load miniconda and activate python virtual environment vqa
# created by conda --create
export PATH="/scratch/alfadlmm/miniconda2/bin:$PATH"
source activate vqa

## RUN YOUR PROGRAM ##
for i in {0..14}
do
	srun python solve.py $i &
done

wait