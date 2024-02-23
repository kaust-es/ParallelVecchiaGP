#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <string>
#include <iomanip>


location *loadXYcsv(const std::string& file_path, int n)
{
    // Check if the file exists by trying to open it
    std::ifstream testFile(file_path);
    if (!testFile) {
        std::cerr << "Error: File " << file_path << " does not exist\n";
        exit(0);
    }
    location *loc = (location *) malloc(sizeof(location));
    loc->x = (double* ) malloc(n * sizeof(double));
    loc->y = (double* ) malloc(n * sizeof(double));
    loc->z = NULL;
    std::ifstream file(file_path);
    std::string line;
    int i = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        loc->x[i] = std::stod(token);
        std::getline(ss, token, ',');
        loc->y[i] = std::stod(token);
        // std::getline(ss, token, ',');
        // loc->z[i] = std::stod(token);
        ++i;
    }
    file.close();
    return loc;
}

template <class T>
void loadObscsv(const std::string& file_path, int n, T* obs)
{
    // Check if the file exists by trying to open it
    std::ifstream testFile(file_path);
    if (!testFile) {
        std::cerr << "Error: File " << file_path << " does not exist\n";
        exit(0);
    }
    // T* obs = (double* ) malloc(n * sizeof(double));
    std::ifstream file(file_path);
    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        T value;
        if (ss >> value) {
            obs[i] = value;
            ++i;
        }
    }
    file.close();
}


int createDirectoryIfNotExists(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return 0; // Directory already exists
        } else {
            return -1; // Path exists but is not a directory
        }
    } else {
        if (mkdir(path, 0777) == 0) {
            return 0; // Directory created successfully
        } else {
            return -1; // Failed to create the directory
        }
    }
}

void createLogFile(kblas_opts &opts)
{
    const char *log_dir = "./log";

    int result = createDirectoryIfNotExists(log_dir);

    if (result == 0) {
        printf("Directory exists or was created successfully.\n");
    } else {
        printf("Failed to create the directory.\n");
    }
}

template<class T>
void saveLogFileSum(int iterations, std::vector<T> theta, double max_llh, double whole_time,  kblas_opts &opts) {

    std::string file_path;
    if (opts.perf == 1){
        // only matern is enough
        file_path = "./log/locs_" + std::to_string(opts.num_loc) + "_" \
                            + "cs_" + std::to_string(opts.vecchia_cs) + "_" \
                            + "seed_" + std::to_string(opts.seed) + "_" \
                            + "kernel_" + std::to_string(opts.sigma) + ":" \
                                + std::to_string(opts.beta) + ":" \
                                + std::to_string(opts.nu);
    }else{
        file_path = "./log/locs_" + std::to_string(opts.num_loc) + "_" \
                            + "cs_" + std::to_string(opts.vecchia_cs);
    }
    if (opts.mortonordering) file_path = file_path + "_morton";
    else if (opts.randomordering) file_path = file_path + "_random";

    // Print the log message to the log file using printf
    printf("Total Number of Iterations = %d \n", iterations);
    printf("Total Optimization Time = %lf secs \n", whole_time);
    // matern + power exponential kernel 
    if (opts.kernel == 1 || opts.kernel == 2){
        printf("Model Parameters (Variance, range, smoothness): (%.8f, %.8f, %.8f) -> Loglik: %.18f \n",\
                theta[0], theta[1], theta[2], max_llh);
        std::ofstream outfile(file_path);

        // Write the headers for the CSV file
        outfile << "Iterations, variance, range, smoothness, log-likelihood, ordering" << std::endl;
        // Write the log data to the CSV file
        if (opts.mortonordering){
            outfile << iterations << ", " \
                    << theta[0] << ", " << theta[1] << ", " << theta[2] << ", " \
                    << std::setprecision(std::numeric_limits<double>::max_digits10) << max_llh << ", morton" << std::endl;
        }else if (opts.randomordering)
        {
            outfile << iterations << ", " \
                    << theta[0] << ", " << theta[1] << ", " << theta[2] << ", " \
                    << std::setprecision(std::numeric_limits<double>::max_digits10) << max_llh << ", random" << std::endl;
        }
        outfile.close();
    }else if (opts.kernel == 3){
        printf("Model Parameters (Variance, range, smoothness, nugget): (%.8f, %.8f, %.8f, %.8f) -> Loglik: %.18f \n",\
                theta[0], theta[1], theta[2], theta[3], max_llh);
        std::ofstream outfile(file_path);

        // Write the headers for the CSV file
        outfile << "Iterations, variance, range, smoothness, nugget, log-likelihood, ordering" << std::endl;
        // Write the log data to the CSV file
        if (opts.mortonordering){
            outfile << iterations << ", " \
                    << theta[0] << ", " << theta[1] << ", " << theta[2] << ", "  << theta[3] << ", " \
                    << std::setprecision(std::numeric_limits<double>::max_digits10) << max_llh << ", morton" << std::endl;
        }else if (opts.randomordering)
        {
            outfile << iterations << ", " \
                    << theta[0] << ", " << theta[1] << ", " << theta[2] << ", "   << theta[3] <<  ", " \
                    << std::setprecision(std::numeric_limits<double>::max_digits10) << max_llh << ", random" << std::endl;
        }
        outfile.close();
    }
}
#endif