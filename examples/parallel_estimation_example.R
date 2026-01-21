# ./bin/examples/Example_Parallel_Vecchia_Estimation 
#   --N=2000 
#   --iTheta=1.5:0.1:0.5 
#   --kernel=univariate_matern_stationary 
#   --performance 
#   --conditioning_size=300 
#   --knn 
#   --block_count=1 
#   --seed=0 
#   --max_mle_iterations=1 
#   --ncores=40 
#   --VecchiaType=parallel 
#   --permutation=random 
#   --gpus=1

library(Vecchia)

cat("=== Parallel Vecchia Estimation Example ===\n")
cat("Replicating C++ command with all parameters\n\n")

# ============================================================================
# Step 1: Load or Generate Data
# ============================================================================

cat("Step 1: Loading data with parallel Vecchia...\n")
cat("Parameters:\n")
cat("  --N=2000\n")
cat("  --block_count=1\n")
cat("  --iTheta=1.5:0.1:0.5\n")
cat("  --kernel=univariate_matern_stationary\n")
cat("  --seed=0\n")
cat("  --VecchiaType=parallel\n")
cat("  --permutation=random\n")
cat("  --conditioning_size=300\n")
cat("  --ncores=40\n")
cat("  --gpus=1\n")
cat("  --knn (R wrapper sets is_knn=true to match C++ behavior)\n")
cat("  --performance (Note: performance flag not exposed in R wrapper)\n\n")

# Load data
data_result <- load_data(
    vecchia_type = "parallel",                 # --VecchiaType=parallel
    kernel = "univariate_matern_stationary",   # --kernel=univariate_matern_stationary
    initial_theta = c(1.5, 0.1, 0.5),          # --iTheta=1.5:0.1:0.5
    distance_matrix = "euclidean",             # Default distance metric
    problem_size = 2000,                       # --N=2000
    seed = 0,                                   # --seed=0
    block_count = 1,                           # --block_count=1 (number of blocks)
    dimension = "2D",                          # For parallel Vecchia, use "2D" or "3D"
    data_path = "",                            # Empty means generate new data
    conditioning_size = 300,                   # --conditioning_size=300
    ncores = 40,                               # --ncores=40
    permutation = "random"                     # --permutation=random
)

# Extract data
x_coords <- data_result$x
y_coords <- data_result$y
measurements <- data_result$m

cat("Generated", length(x_coords), "data points\n")
cat("X range:", range(x_coords), "\n")
cat("Y range:", range(y_coords), "\n")
cat("Measurements range:", range(measurements), "\n\n")

# ============================================================================
# Step 2: Estimate Parameters (Parallel Vecchia MLE)
# ============================================================================

cat("Step 2: Estimating parameters with parallel Vecchia MLE...\n")
cat("Parameters:\n")
cat("  --max_mle_iterations=1\n")
cat("  --tolerance=6 (default)\n")
cat("  --conditioning_size=300\n\n")

# Perform MLE estimation with parallel Vecchia
model_result <- model_data(
    vecchia_type = "parallel",                 # --VecchiaType=parallel
    kernel = "univariate_matern_stationary",   # --kernel=univariate_matern_stationary
    distance_matrix = "euclidean",             # Default distance metric
    lb = c(0.01, 0.01, 0.01),                  # Lower bounds (variance, range, smoothness) - matches C++ defaults
    ub = c(3.0, 3.0, 3.0),                     # Upper bounds (variance, range, smoothness) - matches C++ defaults
    tol = 6,                                    # --tolerance=6 (exponent, means 1e-6)
    mle_itr = 1,                               # --max_mle_iterations=1
    block_count = 1,                           # --block_count=1 (number of blocks)
    dimension = "2D",                          # For parallel Vecchia, use "2D" or "3D"
    data = data_result,                        # Pass data_result to reuse VecchiaGBData and hardware from load_data
    initial_theta = c(1.5, 0.1, 0.5),          # --iTheta=1.5:0.1:0.5
    conditioning_size = 300,                   # --conditioning_size=300
    ncores = 40,                               # --ncores=40
    permutation = "random"                     # --permutation=random
)

# Extract results
log_likelihood <- model_result$log_likelihood
estimated_theta <- model_result$estimated_theta

cat("Optimal log-likelihood:", log_likelihood, "\n")
cat("Estimated theta length:", length(estimated_theta), "\n")
cat("Estimated theta:", paste(estimated_theta, collapse = ", "), "\n\n")

cat("=== Example completed ===\n")

