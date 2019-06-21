1. qsub -I -q k40 -l walltime=12:00:00,nodes=1:ppn=20
    * queue the job

2. singularity shell -B /project,/work --nv /home/admin/singularity/pytorch-1.0.0-dockerhub-v3.simg
    * initializes a singularity shell that will allow pytorch to be run on an os that cannot run pytorch otherwise

3. unset PYTHONPATH
4. unset PYTHONHOME
    * 2 commands that remove any prior python paths so that the virtual environment is used
    * must be done everytime before an environment is created or activated

5. conda create -n pytorch python=3.7
    * creates a virtual conda environment named 'pytorch.' can be skipped if an environment has already been made

6. source activate pytorch
    * activates the virtual conda environment. depending on the conda version, 'source' may be replaced by 'conda'

7. conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    * installs pytorch and torchvision into the environment. can be skipped if pytorch is already installed in the environment

8. export LD_LIBRARY_PATH=/usr/local/onnx/onnx:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib64:/.singularity.d/libs
    * fixes an error: "undefined symbol: _ZTIN2at11TypeDefaultE....."

9. cd ../../work/jfeins1/bionoi_autoencoder_modified
    * cd into directory containing the autoencoder.py

10. python autoencoder_general.py -data_dir /work/jfeins1/bae-test-122k/ -model_file ./log/conv1x1-120k-batch512.pt -style conv_1x1 -batch_size 512
    * the command to run the autoencoder file. 
    
***You must have a .condarc file in your home directory with the paths to your anaconda environments and packages.  Note that .condarc is a hidden file. Must be formatted as below.
```
envs_dirs:
- /work/jfeins1/anaconda3/envs
pkgs_dirs:
- /work/jfeins1/anaconda3/pkgs 
```