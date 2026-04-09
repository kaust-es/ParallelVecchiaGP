# Example R script for prediction-only mode (matching C++ predict.sh)
# This script performs prediction using external files from simu_ds
# Replicates: ./bin/examples/Example_Parallel_Vecchia_Estimation_Prediction
#   --train_locs=... --test_locs=... --train_data=... --test_data=...
#   --itheta=... --kernel=UnivariateMaternStationary --conditioning_size=...
#   --block_count=200 --cores=20 --vecchiaType=block --permutation=random
#   --gpus=0 --dim=2D
# (Estimation is skipped when train/test files are provided - prediction-only mode)

library(Vecchia)

cat("=== Block Vecchia Prediction-Only Mode ===\n")
cat("Using external files from simu_ds (matching predict.sh)\n\n")

# ============================================================================
# Prediction-Only Mode (using external files)
# ============================================================================

cat("Prediction-only mode: Using external files from simu_ds\n")
cat("Parameters (matching predict.sh):\n")
cat("  --train_locs, --test_locs, --train_data, --test_data (file paths)\n")
cat("  --itheta (theta for prediction)\n")
cat("  --kernel=UnivariateMaternStationary\n")
cat("  --conditioning_size=300\n")
cat("  --block_count=200\n")
cat("  --ncores=20\n\n")

beta_str <- "0.014290"  # --itheta beta (range)
nu_str <- "2.500000"    # --itheta nu (smoothness)
i <- 1                   # File number (1-50)

data_folder <- paste0("./simu_ds/prediction_data_", beta_str, "_", nu_str)
train_locs_file <- paste0(data_folder, "/LOC_", i, "_train.csv")
test_locs_file <- paste0(data_folder, "/LOC_", i, "_test.csv")
train_data_file <- paste0(data_folder, "/Z1_", i, "_train.csv")
test_data_file <- paste0(data_folder, "/Z1_", i, "_test.csv")

# Check if files exist
if (!file.exists(train_locs_file) || !file.exists(test_locs_file) || 
    !file.exists(train_data_file) || !file.exists(test_data_file)) {
    stop(paste("Prediction data files not found in:", data_folder))
}

cat("Using prediction data files:\n")
cat("  Train locations:", train_locs_file, "\n")
cat("  Test locations:", test_locs_file, "\n")
cat("  Train data:", train_data_file, "\n")
cat("  Test data:", test_data_file, "\n\n")

# ============================================================================
# Step 1: Load Data (Initialize Hardware and Cluster Test Locations)
# ============================================================================

cat("Step 1: Initializing hardware (via load_data)...\n")
cat("Note: For prediction-only mode, this initializes hardware only.\n")
cat("The actual train/test data loading and clustering will happen in predict_data() using file paths.\n\n")

data_result <- load_data(
    vecchia_type = "block",                    # --vecchiaType=block
    kernel = "univariate_matern_stationary",   # --kernel=univariate_matern_stationary
    initial_theta = c(1.5, 0.1, 0.5),          # Initial theta (will be overridden by estimated_theta in predict_data)
    distance_matrix = "euclidean",             # Default distance metric
    problem_size = 50,                         # --N=50 (matching predict_data)
    seed = 1,                                   # --seed=1 (matching C++ command)
    block_count = 200,                          # --block_count=200 (number of blocks)
    dimension = "2D",                          # --dim=2D
    data_path = "",                            # Empty means generate dummy data (we won't use it)
    conditioning_size = 300,                   # --conditioning_size=300
    ncores = 20,                               # --ncores=20
    permutation = "random"                     # --permutation=random
)

cat("Hardware initialized.\n\n")

# ============================================================================
# Step 2: Perform Prediction
# ============================================================================

sigma_str <- "1.5"  # variance
prediction_theta <- c(as.numeric(sigma_str), as.numeric(beta_str), as.numeric(nu_str))  # --itheta

cat("Step 2: Performing prediction with block Vecchia...\n")
cat("Parameters:\n")
cat("  --train_locs, --test_locs, --train_data, --test_data (file paths)\n")
cat("  --itheta=", paste(prediction_theta, collapse=":"), "\n\n")

predicted_values <- predict_data(
    vecchia_type = "block",                    # --vecchiaType=block
    kernel = "univariate_matern_stationary",   # --kernel=UnivariateMaternStationary
    distance_matrix = "euclidean",             # --distance_metric=euclidean
    estimated_theta = prediction_theta,        # --itheta=1.5,0.014290,2.500000
    block_count = 200,                          # --block_count=200 (number of blocks)
    dimension = "2D",                           # --dim=2D
    train_data = train_data_file,              # --train_data (file path)
    test_data = test_data_file,                # --test_data (file path)
    train_locs = train_locs_file,              # --train_locs (file path)
    test_locs = test_locs_file,                # --test_locs (file path)
    conditioning_size = 300,                   # --conditioning_size=300
    ncores = 20,                                # --cores=20
    permutation = "random",                    # --permutation=random
    seed = 1,                                   # --seed=1 (matching C++ command)
    problem_size = 50                           # --N=50 (matching C++ command)
)

cat("Prediction completed:\n")
cat("  Number of predictions:", length(predicted_values), "\n")
if (length(predicted_values) > 0 && any(predicted_values != 0)) {
    cat("  Predicted values range:", sprintf("[%.4f, %.4f]", 
        min(predicted_values), max(predicted_values)), "\n")
    cat("  Mean prediction:", sprintf("%.4f", mean(predicted_values)), "\n")
    cat("  Std dev:", sprintf("%.4f", sd(predicted_values)), "\n")
} else {
    cat("  Note: Prediction may return placeholder values.\n")
}
cat("\n")

cat("=== Prediction-only mode completed ===\n")

