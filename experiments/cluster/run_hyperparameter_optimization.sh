#!/bin/bash
#CCS -N OrderingOfImageObjectsHP_GPU
#CCS --res=rset=1:ncpus=2:mem=30g:vmem=60g:gpus=1:tesla=f
#CCS -mea
#CCS -t 2d
#CCS -M fweiland@mail.upb.de

export IORANK_TIME_LIMIT=172800

module add singularity
cd /upb/scratch/departments/pc2/groups/HPC-LTH/
export PYTHONPATH=/upb/scratch/departments/pc2/groups/HPC-LTH/code
singularity run --nv singularity/pytorch.sif python code/iorank/experiments/experiment_runner.py --config conf/conf_gpu.yml
