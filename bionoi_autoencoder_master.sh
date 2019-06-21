#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -q k40
#PBS -N bionoi_autoencoder
#PBS -A loni_bionoi01
#PBS -j oe

singularity shell -B /project,/work --nv /home/admin/singularity/pytorch-1.0.0-dockerhub-v3.simg
unset PYTHONPATH
unset PYTHONHOME
source activate pytorch
export LD_LIBRARY_PATH=/usr/local/onnx/onnx:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib64:/.singularity.d/libs
cd ~
cd ../../work/jfeins1/bionoi_autoencoder_modified
python autoencoder_general.py -data_dir /work/jfeins1/bae-images/ -model_file ./log/conv1x1-4M-batch512.pt -style conv_1x1 -batch_size 512