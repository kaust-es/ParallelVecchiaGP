#-----------------------------------------
ifndef _USE_OPENBLAS_
_USE_MKL_?=TRUE
endif
#-----------------------------------------
_USE_MAGMA_?=TRUE
#-----------------------------------------
#specify cuda directory
_CUDA_ROOT_=$(CUDA_HOME)
_CUDA_ARCH_ ?= 70

# specify compilers
CXX ?= g++
CC ?= gcc
NVCC=$(_CUDA_ROOT_)/bin/nvcc
#customize the name of the static kblas library
LIB_KBLAS_NAME=kblas-gpu

#-----------------------------------------
NVOPTS = -ccbin $(CXX) --compiler-options -fno-strict-aliasing
COPTS = -fopenmp

NVOPTS_3 = -DTARGET_SM=$(_CUDA_ARCH_) -allow-unsupported-compiler -arch sm_$(_CUDA_ARCH_) -Xcompiler -fopenmp

#-----------------------------------------
ifdef _DEBUG_
  COPTS += -g -Xcompiler -rdynamic
  NVOPTS += -G -g -lineinfo
else
  COPTS += -O3
  NVOPTS += -O3
endif

#-----------------------------------------
ifdef _USE_MAGMA_
  COPTS += -DUSE_MAGMA
  _MAGMA_ROOT_?=$(HOME)/scratch/codes/magma-2.5.2
  NVOPTS += -DUSE_MAGMA
endif
#-----------------------------------------
ifdef _USE_MKL_
  COPTS += -DUSE_MKL
  NVOPTS += -DUSE_MKL
  _MKL_ROOT_?=${MKLROOT}
endif

# the source file
VECCHIA_SOURCE=./src
VECCHIA_HEADER=./include


# include and lib paths
INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I./include
INCLUDES+= -I${_CUDA_ROOT_}/include
INCLUDES+= -I${_KBLAS_ROOT_}/include -I${_KBLAS_ROOT_}/src
INCLUDES+= -I${_NLOPT_ROOT_}/include
# INCLUDES+= -I${_GSL_ROOT_}/include

ifdef _USE_MAGMA_
	INCLUDES+= -I$(_MAGMA_ROOT_)/include
endif
ifdef _USE_MKL_
	INCLUDES+= -I${_MKL_ROOT_}/include
endif
ifdef _USE_OPENBLAS_
	INCLUDES+= -I${_OPENBLAS_INCLUDE_}
endif

LIB_PATH=
LIB_PATH+= -L${_CUDA_ROOT_}/lib64
LIB_PATH+= -L${_KBLAS_ROOT_}/lib
LIB_PATH+= -L${_NLOPT_ROOT_}/lib
# LIB_PATH+= -L${_GSL_ROOT_}/lib

ifdef _USE_MAGMA_
	LIB_PATH+= -L${_MAGMA_ROOT_}/lib
endif
ifdef _USE_MKL_
	LIB_PATH+= -L${_MKL_ROOT_}/lib/intel64
endif
ifdef _USE_OPENBLAS_
	LIB_PATH+= -L${_OPENBLAS_LIB_}
endif

# libraries to link against
LIB= -lm -l${LIB_KBLAS_NAME}
LIB+= -lnlopt  
#-lgsl
ifdef _USE_MAGMA_
	LIB+= -lmagma -lcusparse
endif
LIB+= -lcublas -lcudart
ifdef _USE_MKL_
	# LIB+= -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl
	LIB+= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	# LIB+= -mkl=sequential
	# LIB+= -mkl=parallel
endif
ifdef _USE_OPENBLAS_
	LIB+= -lopenblas
endif
LIB+= -lgomp
LIB+= -lstdc++

KBLAS_LIB=${_KBLAS_ROOT_}/lib/lib${LIB_KBLAS_NAME}.a

OBJ_DIR=./obj
BIN_DIR=./bin

# Corrected directory creation using variables
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

_VECCHIA_TEST_?=TRUE
# $(info $$_VECCHIA_TEST_ is [${_VECCHIA_TEST_}])
ifdef _VECCHIA_TEST_
include $(VECCHIA_SOURCE)/Makefile
endif

all: $(EXE_BATCH)

$(OBJ_DIR)/testing_helper.o: $(VECCHIA_SOURCE)/testing_helper.cu
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INCLUDES) $(NVCCFLAGS)  -c $< -o $@

# dotproduct operations
$(OBJ_DIR)/gpukernels.o: $(VECCHIA_SOURCE)/gpukernels.cu
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INCLUDES) $(NVCCFLAGS)  -c $< -o $@

$(OBJ_DIR)/ckernel.o: $(VECCHIA_SOURCE)/ckernel.cpp
	$(CXX) $(COPTS) $(INCLUDES) -c $< -o $@

$(EXE_BATCH): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(KBLAS_LIB) $(OBJ_DIR)/testing_helper.o $(OBJ_DIR)/ckernel.o $(OBJ_DIR)/gpukernels.o
	$(CC) $(COPTS) $(OBJ_DIR)/testing_helper.o $(OBJ_DIR)/ckernel.o $(OBJ_DIR)/gpukernels.o $< -o $@ $(LIB_PATH) $(LIB)

clean:
	rm -f $(OBJ_DIR)/*.o $(EXE_BATCH)
