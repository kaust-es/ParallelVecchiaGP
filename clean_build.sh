#! /bin/sh
# Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file configure
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Khaled
# @date 2025-09-29

# Define variables.
verbose=""
num_proc="-j $(nproc)"  # Use the number of available processors by default.

# Parse command-line arguments.
while getopts "vj:h" opt; do
  case $opt in
    v)
      verbose="VERBOSE=1"
      echo "Using verbose mode"
      ;;
    j)
      num_proc="-j $OPTARG"
      echo "Using $OPTARG threads to build"
      ;;
    h)
      # Print help information and exit.
      echo "Usage: $(basename "$0") [-v] [-j <thread_number>] [-h]"
      echo "Clean and build the Vecchia software package."
      echo ""
      echo "Options:"
      echo "  -v                 Use verbose output."
      echo "  -h                 Show this help message."
      echo "  -j <thread_number> Build with a specific number of threads."
      exit 0
      ;;
    *)
      # Print an error message and exit.
      echo "Invalid flag. Use the -h flag for help."
      exit 1
      ;;
  esac
done

# Change to the bin directory, or exit if it doesn't exist.
cd bin/ || {
  echo "Error: bin directory not found."
  exit 1
}

# Clean the directory and build the code with the specified options.
cmake --build . "$num_proc" $verbose
