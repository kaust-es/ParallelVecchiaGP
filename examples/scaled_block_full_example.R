# mpirun -n 1 ./bin/examples/Example_Parallel_Vecchia_Estimation_Prediction 
#   --N=1000 
#   --block_size=100 
#   --distance_scale=0.05:0.05:0.1:1.0:1.0:1.0:1.0:1.0 
#   --nn_multiplier=500 
#   --iTheta=1.0:0.001 
#   --kernel=univariate_matern_stationary 
#   --seed=0 
#   --max_mle_iterations=1 
#   --ncores=20 
#   --VecchiaType=scaled_block 
#   --permutation=morton 
#   --gpus=0 
#   --conditioning_size=200 
#   --dim=8 
#   --kernel_type=Matern72 
#   --tolerance=6
# This script performs both estimation and prediction

library(Vecchia)

cat("=== Scaled Block Vecchia Estimation Example ===\n")
cat("Replicating C++ command with all parameters\n\n")

# ============================================================================
# Step 1: Load or Generate Data
# ============================================================================

cat("Step 1: Loading data with scaled block Vecchia...\n")
cat("Parameters:\n")
cat("  --N=1000\n")
cat("  --block_size=100\n")
cat("  --iTheta=1.0:0.001\n")
cat("  --kernel=univariate_matern_stationary\n")
cat("  --seed=0\n")
cat("  --VecchiaType=scaled_block\n")
cat("  --permutation=morton\n")
cat("  --conditioning_size=200\n")
cat("  --ncores=20\n")
cat("  --gpus=0\n")
cat("  --dim=8\n")
cat("  --kernel_type=Matern72\n")
cat("  --distance_scale=0.05:0.05:0.1:1.0:1.0:1.0:1.0:1.0\n")
cat("  --nn_multiplier=500\n")
cat("  --knn (R wrapper sets is_knn=true to match C++ behavior)\n\n")


data_result <- load_data(
    vecchia_type = "scaled_block",            # --VecchiaType=scaled_block
    kernel = "univariate_matern_stationary",  # --kernel=univariate_matern_stationary
    initial_theta = c(1.0, 0.001),           # --iTheta=1.0:0.001 (variance, nugget)
    distance_matrix = "euclidean",
    problem_size = 1000,                      # --N=1000
    seed = 0,                                 # --seed=0
    block_size = 100,                        # --block_size=100
    dimension = "8",                          # --dim=8 (for scaled_block, use integer as string)
    data_path = "",                           # Empty for synthetic data
    distance_scale = c(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),  # --distance_scale=0.05:0.05:0.1:1.0:1.0:1.0:1.0:1.0
    nn_multiplier = 500,                      # --nn_multiplier=500
    conditioning_size = 200,                  # --conditioning_size=200
    ncores = 20,                              # --ncores=20
    permutation = "morton",                    # --permutation=morton
    kernel_type = "Matern72"                  # --kernel_type=Matern72
)

# Extract components
measurements <- data_result$m
x_coords <- data_result$x
y_coords <- data_result$y

cat("Generated", length(measurements), "data points\n")
cat("X range:", range(x_coords), "\n")
cat("Y range:", range(y_coords), "\n")
cat("Measurements range:", range(measurements), "\n\n")

# ============================================================================
# Step 2: Estimate Parameters (Scaled Block Vecchia MLE)
# ============================================================================

cat("Step 2: Estimating parameters with scaled block Vecchia MLE...\n")
cat("Parameters:\n")
cat("  --max_mle_iterations=1\n")
cat("  --tolerance=6\n")
cat("  --conditioning_size=200\n\n")

model_result <- model_data(
    vecchia_type = "scaled_block",             # --VecchiaType=scaled_block
    kernel = "univariate_matern_stationary",   # --kernel=univariate_matern_stationary
    distance_matrix = "euclidean",             # Default distance metric
    lb = c(0.001, 1e-6, 5e-5, 5e-5, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001),  # Lower bounds (10 values for 8D)
    ub = c(10, 0.01, 0.5, 0.5, 1, 10, 10, 10, 10, 10),  # Upper bounds (10 values for 8D)
    tol = 6,                                    # --tolerance=6 (exponent, means 1e-6)
    mle_itr = 1,                               # --max_mle_iterations=1
    block_size = 100,                           # --block_size=100
    dimension = "8",                            # --dim=8 (8D for scaled_block)
    data = data_result,                        # Pass data_result to reuse VecchiaGBData and hardware from load_data
    initial_theta = c(1.0, 0.001),            # --iTheta=1.0:0.001 (variance, nugget only)
    distance_scale = c(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),  # --distance_scale=0.05:0.05:0.1:1.0:1.0:1.0:1.0:1.0
    nn_multiplier = 500,                       # --nn_multiplier=500
    conditioning_size = 200,                   # --conditioning_size=200
    ncores = 20,                               # --ncores=20
    permutation = "morton",                   # --permutation=morton
    kernel_type = "Matern72"                   # --kernel_type=Matern72
)

# Extract results
log_likelihood <- model_result$log_likelihood
estimated_theta <- model_result$estimated_theta

cat("Optimal log-likelihood:", log_likelihood, "\n")
cat("Estimated theta length:", length(estimated_theta), "\n")
cat("Estimated theta:", paste(estimated_theta, collapse = ", "), "\n\n")
cat("Expected theta format: [variance, nugget, distance_scale[0], ..., distance_scale[7]]\n")
cat("  (10 values total for 8D scaled_block)\n\n")

# ============================================================================
# Step 3: Perform Prediction (Scaled Block Vecchia)
# ============================================================================

cat("Step 3: Performing prediction with scaled block Vecchia...\n")
cat("Parameters:\n")
cat("  Using estimated_theta from Step 2\n")
cat("  --conditioning_size=200\n")
cat("  --block_size=100\n\n")

# Perform prediction using the estimated theta from Step 2
predicted_values <- predict_data(
    vecchia_type = "scaled_block",             # --VecchiaType=scaled_block
    kernel = "univariate_matern_stationary",   # --kernel=univariate_matern_stationary
    distance_matrix = "euclidean",             # Default distance metric
    estimated_theta = estimated_theta,         # Use estimated theta from Step 2
    block_size = 100,                           # --block_size=100
    dimension = "8",                            # --dim=8 (8D for scaled_block)
    train_data = data_result,                  # Pass data_result to reuse VecchiaGBData and hardware from load_data
    test_data = NULL,                          # NULL - test locations are already in data_result from load_data
    distance_scale = c(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),  # --distance_scale=0.05:0.05:0.1:1.0:1.0:1.0:1.0:1.0
    nn_multiplier = 500,                       # --nn_multiplier=500
    conditioning_size = 200,                   # --conditioning_size=200
    ncores = 20,                               # --ncores=20
    permutation = "morton",                   # --permutation=morton
    kernel_type = "Matern72"                   # --kernel_type=Matern72
)

cat("Prediction completed:\n")
cat("  Number of predictions:", length(predicted_values), "\n")
if (length(predicted_values) > 0) {
    cat("  Predicted values range:", sprintf("[%.4f, %.4f]", 
        min(predicted_values), max(predicted_values)), "\n")
    cat("  Mean prediction:", sprintf("%.4f", mean(predicted_values)), "\n")
    cat("  Std dev:", sprintf("%.4f", sd(predicted_values)), "\n")
} else {
    cat("  Note: No predictions returned.\n")
}
cat("\n")

cat("=== Example completed ===\n")