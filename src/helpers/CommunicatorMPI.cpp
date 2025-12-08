
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file CommunicatorMPI.cpp
 * @brief Defines the CommunicatorMPI class for MPI rank communication.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <helpers/CommunicatorMPI.hpp>

using namespace vecchia::helpers;

CommunicatorMPI *CommunicatorMPI::GetInstance() {
    if (mpInstance == nullptr) {
        mpInstance = new CommunicatorMPI();
    }
    return mpInstance;
}

int CommunicatorMPI::GetRank() const {
#ifdef USE_MPI
    if (!mIsHardwareInitialized) {
        return 0;
    } 
    else {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
#else
    return 0;
#endif
}

void CommunicatorMPI::SetHardwareInitialization() {
    mIsHardwareInitialized = true;
}

void CommunicatorMPI::RemoveHardwareInitialization() {
    mIsHardwareInitialized = false;
}

CommunicatorMPI *CommunicatorMPI::mpInstance = nullptr;
