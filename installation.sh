#!/bin/bash
# load necessary modules
module load intel/2022
module load gcc/10.2.0
module load cuda/11.4
# specify the CUDA path
export CUDA_ROOT=$CUDA_HOME
export CUDADIR=$CUDA_HOME
export _CUB_DIR_=$CUDA_HOME
echo $MKLROOT
# download the MAGAMA 2.6.0
wget http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.6.0.tar.gz
tar -xzf magma-2.6.0.tar.gz
cd magma-2.6.0
cp make.inc-examples/make.inc.mkl-gcc make.inc
# specify your arch of GPU, or Ampere, (Hopper for MAGMA 2.7.2 and later)
export GPU_TARGET=Volta
export _MAGMA_ROOT_=$(pwd)
make -j
# add the library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your-path/magma-2.6.0/lib
export LIBRARY_PATH=$LIBRARY_PATH:/your-path/magma-2.6.0/lib
# install the kblas-gpu
git clone https://github.com/ecrc/kblas-gpu.git
export _KBLAS_ROOT_=$(pwd)
make -j
