#!/bin/bash
#CCS -N OrderingOfImageObjectsGPU
#CCS --res=rset=1:ncpus=2:mem=30g:vmem=60g:gpus=1:tesla=f
#CCS -mea
#CCS -t 12h
#CCS -M fweiland@mail.upb.de

module add singularity
cd /upb/scratch/departments/pc2/groups/HPC-LTH/
export PYTHONPATH=/upb/scratch/departments/pc2/groups/HPC-LTH/code
singularity run --nv singularity/pytorch.sif python code/iorank/experiments/component_experiment_runner.py --config conf/conf_gpu.yml
