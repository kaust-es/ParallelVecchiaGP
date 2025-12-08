
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file VecchiaGBData.hpp
 * @brief Contains the definition of the VecchiaGBData class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_DATA_HPP
#define VECCHIAGP_DATA_HPP

#include <magma_v2.h>
#include <memory>

#include <data-units/Locations.hpp>
#include <data-units/BlockInfo.hpp>
#include <utilities/TimingData.hpp>

// Forward declaration to avoid circular dependency
namespace vecchia { namespace clustering {
    template<typename T> struct ClusteringResult;
}}

/**
 * @Class VecchiaGBData
 * @brief  Manages geo-statistical data with functions for location and descriptor manipulation
 * @tparam T Data Type: float or double
 */
template<typename T>
class VecchiaGBData {

public:
    /**
     * @brief Constructor for VecchiaGBData.
     * @param[in] aSize The size of the data.
     * @param[in] aDimension The dimension of the data.
     *
     */
    VecchiaGBData(const int &aSize, const vecchia::common::Dimension &aDimension, const int &aBlockSize);

    /**
     * @brief Constructor for VecchiaGBData.
     * @param[in] aSize The size of the data.
     * @param[in] aDimension The dimension of the data.
     *
     */
    VecchiaGBData(const int &aSize, const std::string &aDimension);

    /**
     * @brief Default constructor for VecchiaGBData.
     *
     */
    VecchiaGBData() = default;

    /**
     * @brief Destructor for VecchiaGBData.
     *
     */
    ~VecchiaGBData();

    /**
     * @brief Get the locations.
     * @return Pointer to the Locations object.
     *
     */
    vecchia::dataunits::Locations<T> *GetLocations();

    /**
     * @brief Set the locations.
     * @param[in] aLocation Pointer to the Locations object.
     * @return void
     *
     */
    void SetLocations(vecchia::dataunits::Locations<T> &aLocation);

    /**
     * @brief Get the centroids locations.
     * @return Pointer to the Locations object.
     *
     */
    vecchia::dataunits::Locations<T> *GetCentroidsLocations();

    /**
     * @brief Set the centroids locations.
     * @param[in] aCentroidsLocations Pointer to the Locations object.
     * @return void
     *
     */
    void SetCentroidsLocations(vecchia::dataunits::Locations<T> &aCentroidsLocations);

    /**
     * @brief Get the conditioning locations.
     * @return Pointer to the Locations object.
     *
     */
    vecchia::dataunits::Locations<T> *GetConditioningLocations();

    /**
     * @brief Set the conditioning locations.
     * @param[in] aConditioningLocations Pointer to the Locations object.
     * @return void
     *
     */
    void SetConditioningLocations(vecchia::dataunits::Locations<T> &aConditioningLocations);

    /**
     * @brief Get the premIndex.
     * @return Pointer to the premIndex object.
     *
     */
    int *GetPremIndex();

    /**
     * @brief Setter for the number of performed MLE iterations.
     * @param[in] aMleIterations number of performed MLE iterations.
     * @return void
     *
     */
    void SetMleIterations(const int &aMleIterations);

    /**
     * @brief Get the number of performed MLE iterations.
     * @return Pointer to the DescriptorData object.
     *
     */
    int GetMleIterations();

    /**
     * @brief Get the batch count.
     * @return The batch count.
     *
     */
    int GetBatchCount();

    /**
     * @brief Set the batch count.
     * @param[in] aBatchCount The batch count.
     * @return void
     *
     */
    void SetBatchCount(const int &aBatchCount);

    /**
     * @brief Get the host observations.
     * @return The host observations.
     *
     */
    double *GetHostObservations();

    /**
     * @brief Set the host observations.
     * @param[in] aHostObservations The host observations.
     * @return void
     *
     */
    void SetHostObservations(double *aHostObservations);

        /**
     * @brief Get the host observations.
     * @return The host observations.
     *
     */
     double *GetHostObservationsNew();

     /**
      * @brief Set the host observations.
      * @param[in] aHostObservations The host observations.
      * @return void
      *
      */
     void SetHostObservationsNew(double *aHostObservations);
    /**
     * @brief Get the first cluster count.
     * @return The first cluster count.
     *
     */
    int *GetFirstClusterCount();

    /**
     * @brief Get the batch num accum.
     * @return The batch num accum.
     *
     */
    int *GetBatchNumAccum();

    int *GetBatchNumSquareAccum() {return this->mpBatchNumSquareAccum;};

    /**
     * @brief Set the batch num accum.
     * @param[in] aBatchNumAccum The batch num accum pointer.
     * @return void
     *
     */
    void SetBatchNumAccum(int *aBatchNumAccum);

    void SetBatchNumSquareAccum(int *aBatchNumSquareAccum);
    /**
     * @brief Get the new locations.
     * @return The new locations.
     *
     */
    vecchia::dataunits::Locations<T> *GetNewLocations();

    /**
     * @brief Set the new locations.
     * @param[in] aNewLocations The new locations.
     * @return void
     *
     */
    void SetNewLocations(vecchia::dataunits::Locations<T> &aNewLocations);

    /**
     * @brief Get the batch num.
     * @return The batch num.
     *
     */
    int *GetBatchNum();

    /**
     * @brief Set the batch num.
     * @param[in] aBatchNum The batch num.
     * @return void
     *
     */
    void SetBatchNum(int *aBatchNum);


    // Host memory pointers
    double* GetHostCovariance() { return mpHostCovariance; }
    double* GetHostConditioningCov() { return mpHostCovarianceConditioning; }
    double* GetHostConditioningObs() { return mpHostObservationsConditioning; }
    double* GetHostCrossCov() { return mpHostCovarianceCross; }
    // Host memory setters
    void SetHostCovariance(double* ptr) { mpHostCovariance = ptr; }
    void SetHostConditioningCov(double* ptr) { mpHostCovarianceConditioning = ptr; }
    void SetHostConditioningObs(double* ptr) { mpHostObservationsConditioning = ptr; }
    void SetHostCrossCov(double* ptr) { mpHostCovarianceCross = ptr; }
    // Device memory pointers
    double* GetDeviceCovariance() { return mpDeviceCovariance; }
    double* GetDeviceObservations() { return mpDeviceObservations; }
    double* GetDeviceObservationsCopy() { return mpDeviceObservationsCopy; }
    double* GetDeviceConditioningCov() { return mpDeviceCovarianceConditioning; }
    double* GetDeviceConditioningObs() { return mpDeviceObservationsConditioning; }
    double* GetDeviceCrossCov() { return mpDeviceCovarianceCross; }
    double* GetDeviceCovOffset() { return mpDeviceCovarianceOffset; }
    double* GetDeviceMuOffset() { return mpDeviceMuOffset; }
    
    // Device memory setters
    void SetDeviceCovariance(double* ptr) { mpDeviceCovariance = ptr; }
    
    /**
     * @brief Get the BlockInfo vector for Scaled Block Vecchia
     * @return Reference to the BlockInfo vector
     */
    std::vector<vecchia::dataunits::BlockInfo>& GetBlockInfos() { return mBlockInfos; }
    
    /**
     * @brief Set the BlockInfo vector for Scaled Block Vecchia
     * @param[in] aBlockInfos The BlockInfo vector to store
     */
    void SetBlockInfos(const std::vector<vecchia::dataunits::BlockInfo>& aBlockInfos) { 
        mBlockInfos = aBlockInfos; 
    }
    
    /**
     * @brief Get the BlockInfo vector for test/prediction data
     * @return Reference to the BlockInfo vector
     */
    std::vector<vecchia::dataunits::BlockInfo>& GetBlockInfos_test() { return mBlockInfos_test; }
    
    /**
     * @brief Set the BlockInfo vector for test/prediction data
     * @param[in] aBlockInfos The BlockInfo vector to store
     */
    void SetBlockInfos_test(const std::vector<vecchia::dataunits::BlockInfo>& aBlockInfos) { 
        mBlockInfos_test = aBlockInfos; 
    }
    
    /**
     * @brief Get the clustering result for test/prediction locations
     * @return Pointer to the ClusteringResult (nullptr if not set)
     */
    vecchia::clustering::ClusteringResult<T>* GetTestClusteringResult() { 
        return mpTestClusteringResult.get(); 
    }
    
    /**
     * @brief Set the clustering result for test/prediction locations
     * @param[in] aClusteringResult The clustering result to store (copied)
     */
    void SetTestClusteringResult(const vecchia::clustering::ClusteringResult<T>& aClusteringResult);
    
    void SetDeviceObservations(double* ptr) { mpDeviceObservations = ptr; }
    void SetDeviceObservationsCopy(double* ptr) { mpDeviceObservationsCopy = ptr; }
    void SetDeviceConditioningCov(double* ptr) { mpDeviceCovarianceConditioning = ptr; }
    void SetDeviceConditioningObs(double* ptr) { mpDeviceObservationsConditioning = ptr; }
    void SetDeviceCrossCov(double* ptr) { mpDeviceCovarianceCross = ptr; }
    void SetDeviceCovOffset(double* ptr) { mpDeviceCovarianceOffset = ptr; }
    void SetDeviceMuOffset(double* ptr) { mpDeviceMuOffset = ptr; }
    
    // Batch array pointers (array of pointers for MAGMA batch operations)
    T** GetHostCovarianceArray() { return mpHostCovarianceArray; }
    T** GetDeviceCovarianceArray() { return mpDeviceCovarianceArray; }
    T** GetHostObservationsArray() { return mpHostObservationsArray; }
    T** GetDeviceObservationsArray() { return mpDeviceObservationsArray; }
    T** GetHostObservationsArrayCopy() { return mpHostObservationsArrayCopy; }
    T** GetDeviceObservationsArrayCopy() { return mpDeviceObservationsArrayCopy; }
    
    // Leading dimension arrays
    magma_int_t* GetHostLDA() { return mpHostLDA; }
    magma_int_t* GetHostLDDA() { return mpHostLDDA; }
    magma_int_t* GetDeviceLDA() { return mpDeviceLDA; }
    magma_int_t* GetDeviceLDDA() { return mpDeviceLDDA; }
    
    // Info arrays for error checking
    magma_int_t* GetHostInfo() { return mpHostInfo; }
    magma_int_t* GetDeviceInfo() { return mpDeviceInfo; }
    magma_int_t* GetHostConst1() { return mpHostConst1; }
    magma_int_t* GetDeviceConst1() { return mpDeviceConst1; }
    magma_int_t* GetDeviceBatchNum() { return mpDeviceBatchNum; }
    
    // Result arrays
    double* GetLogDetResults() { return mpHostLogDetResults; }
    double* GetNorm2Results() { return mpHostNorm2Results; }
    
    // Additional setter methods for memory management
    void SetHostCovarianceArray(T** ptr) { mpHostCovarianceArray = ptr; }
    void SetDeviceCovarianceArray(T** ptr) { mpDeviceCovarianceArray = ptr; }
    void SetHostObservationsArray(T** ptr) { mpHostObservationsArray = ptr; }
    void SetDeviceObservationsArray(T** ptr) { mpDeviceObservationsArray = ptr; }
    void SetHostObservationsArrayCopy(T** ptr) { mpHostObservationsArrayCopy = ptr; }
    void SetDeviceObservationsArrayCopy(T** ptr) { mpDeviceObservationsArrayCopy = ptr; }
    
    void SetHostLDA(magma_int_t* ptr) { mpHostLDA = ptr; }
    void SetHostLDDA(magma_int_t* ptr) { mpHostLDDA = ptr; }
    void SetDeviceLDA(magma_int_t* ptr) { mpDeviceLDA = ptr; }
    void SetDeviceLDDA(magma_int_t* ptr) { mpDeviceLDDA = ptr; }
    
    void SetHostInfo(magma_int_t* ptr) { mpHostInfo = ptr; }
    void SetDeviceInfo(magma_int_t* ptr) { mpDeviceInfo = ptr; }
    void SetHostConst1(magma_int_t* ptr) { mpHostConst1 = ptr; }
    void SetDeviceConst1(magma_int_t* ptr) { mpDeviceConst1 = ptr; }
    void SetDeviceBatchNum(magma_int_t* ptr) { mpDeviceBatchNum = ptr; }
    
    void SetLogDetResults(double* ptr) { mpHostLogDetResults = ptr; }
    void SetNorm2Results(double* ptr) { mpHostNorm2Results = ptr; }
    
    /**
     * @brief Get the timing data
     * @return Reference to the timing data structure
     */
    vecchia::utilities::TimingData& GetTimingData() { return mTimingData; }
    
    /**
     * @brief Set the timing data
     * @param[in] aTimingData The timing data to store
     */
    void SetTimingData(const vecchia::utilities::TimingData& aTimingData) { mTimingData = aTimingData; }
    
    // Conditioning-related setters
    void SetHostCovarianceConditioningArray(T** ptr) { mpHostCovarianceConditioningArray = ptr; }
    void SetDeviceCovarianceConditioningArray(T** ptr) { mpDeviceCovarianceConditioningArray = ptr; }
    void SetHostCovarianceCrossArray(T** ptr) { mpHostCovarianceCrossArray = ptr; }
    void SetDeviceCovarianceCrossArray(T** ptr) { mpDeviceCovarianceCrossArray = ptr; }
    void SetHostCovarianceOffsetArray(T** ptr) { mpHostCovarianceOffsetArray = ptr; }
    void SetDeviceCovarianceOffsetArray(T** ptr) { mpDeviceCovarianceOffsetArray = ptr; }
    void SetHostMuOffsetArray(T** ptr) { mpHostMuOffsetArray = ptr; }
    void SetDeviceMuOffsetArray(T** ptr) { mpDeviceMuOffsetArray = ptr; }
    void SetHostObservationsConditioningArray(T** ptr) { mpHostObservationsConditioningArray = ptr; }
    void SetDeviceObservationsConditioningArray(T** ptr) { mpDeviceObservationsConditioningArray = ptr; }
    void SetHostObservationsConditioningArrayCopy(T** ptr) { mpHostObservationsConditioningArrayCopy = ptr; }
    void SetDeviceObservationsConditioningArrayCopy(T** ptr) { mpDeviceObservationsConditioningArrayCopy = ptr; }
    
    void SetHostLDAConditioning(magma_int_t* ptr) { mpHostLDAConditioning = ptr; }
    void SetHostLDDAConditioning(magma_int_t* ptr) { mpHostLDDAConditioning = ptr; }
    magma_int_t* GetHostLDDAConditioning() { return mpHostLDDAConditioning; }
    magma_int_t* GetHostLDAConditioning() { return mpHostLDAConditioning; }
    magma_int_t* GetDeviceLDAConditioning() { return mpDeviceLDAConditioning; }
    void SetDeviceLDAConditioning(magma_int_t* ptr) { mpDeviceLDAConditioning = ptr; }
    void SetDeviceLDDAConditioning(magma_int_t* ptr) { mpDeviceLDDAConditioning = ptr; }
    magma_int_t* GetDeviceLDDAConditioning() { return mpDeviceLDDAConditioning; }
    
    // Conditioning array getters
    T** GetHostCovarianceConditioningArray() { return mpHostCovarianceConditioningArray; }
    T** GetDeviceCovarianceConditioningArray() { return mpDeviceCovarianceConditioningArray; }
    T** GetHostCovarianceCrossArray() { return mpHostCovarianceCrossArray; }
    T** GetDeviceCovarianceCrossArray() { return mpDeviceCovarianceCrossArray; }
    T** GetHostCovarianceOffsetArray() { return mpHostCovarianceOffsetArray; }
    T** GetDeviceCovarianceOffsetArray() { return mpDeviceCovarianceOffsetArray; }
    T** GetHostMuOffsetArray() { return mpHostMuOffsetArray; }
    T** GetDeviceMuOffsetArray() { return mpDeviceMuOffsetArray; }
    T** GetHostObservationsConditioningArray() { return mpHostObservationsConditioningArray; }
    T** GetDeviceObservationsConditioningArray() { return mpDeviceObservationsConditioningArray; }
    T** GetHostObservationsConditioningArrayCopy() { return mpHostObservationsConditioningArrayCopy; }
    T** GetDeviceObservationsConditioningArrayCopy() { return mpDeviceObservationsConditioningArrayCopy; }
    
    void SetHostObservationsConditioningCopy(double* ptr) { mpDeviceObservationsConditioningCopy = ptr; }
    void SetDeviceObservationsConditioningCopy(double* ptr) { mpDeviceObservationsConditioningCopy = ptr; }
    double* GetDeviceObservationsConditioningCopy() { return mpDeviceObservationsConditioningCopy; }

    void SetTotalSizeDeviceObservations(long long value) { mTotalSizeDeviceObservations = value; }
    long long GetTotalSizeDeviceObservations() { return mTotalSizeDeviceObservations; }
    
    // ========== Multi-GPU Parallel Vecchia Getters/Setters ==========
    void SetNumGPUs(int num) { mNumGPUs = num; }
    int GetNumGPUs() { return mNumGPUs; }
    
    void SetBatchCountGPU(int* ptr) { mpBatchCountGPU = ptr; }
    int* GetBatchCountGPU() { return mpBatchCountGPU; }
    
    void SetDotResultH(double** ptr) { mpDotResultH = ptr; }
    double** GetDotResultH() { return mpDotResultH; }
    
    void SetLogdetResultH(double** ptr) { mpLogdetResultH = ptr; }
    double** GetLogdetResultH() { return mpLogdetResultH; }
    
    void SetDeviceC(double** ptr) { mpDeviceC = ptr; }
    double** GetDeviceC() { return mpDeviceC; }
    
    void SetDeviceInfoArray(int** ptr) { mpDeviceInfoArray = ptr; }
    int** GetDeviceInfoArray() { return mpDeviceInfoArray; }
    
    void SetLocationsXXD(double** ptr) { mpLocationsXXD = ptr; }
    double** GetLocationsXXD() { return mpLocationsXXD; }
    
    void SetLocationsYYD(double** ptr) { mpLocationsYYD = ptr; }
    double** GetLocationsYYD() { return mpLocationsYYD; }
    
    void SetLocationsConXXD(double** ptr) { mpLocationsConXXD = ptr; }
    double** GetLocationsConXXD() { return mpLocationsConXXD; }
    
    void SetLocationsConYYD(double** ptr) { mpLocationsConYYD = ptr; }
    double** GetLocationsConYYD() { return mpLocationsConYYD; }
    
private:
    //// Used locations data.
    vecchia::dataunits::Locations<T> *mpLocations = nullptr;
    //// Used centroids data.
    vecchia::dataunits::Locations<T> *mpCentroidsLocations = nullptr;
    //// Used new locations.
    vecchia::dataunits::Locations<T> *mpNewLocations = nullptr;
    //// Used conditioning locations.
    vecchia::dataunits::Locations<T> *mpConditioningLocations = nullptr;
    //// Used number of clusters.
    int mNumberOfClusters = 0;
    //// Used batch count.
    int mBatchCount = 0;
    
    //// Used batch number array.
    int *mpBatchNum = nullptr;
    //// Used first cluster count.
    int *mpBatchNumAccum = nullptr;
    //// Used batch number square accum.
    int* mpBatchNumSquareAccum = nullptr;

    //// Used premIndex data.
    int *mpPremIndex = nullptr;
    //// Used cluster number data.
    int *mpClusterNum = nullptr;
    //// Current number of performed MLE iterations.
    int mMleIterations = 0;
    //// Used host observations.
    double *mpHostObservations = nullptr;
    double *mpHostObservationsNew = nullptr;
    //// Used first cluster count.
    int *mpFirstClusterCount = nullptr;

    // Host memory
    // TODO: There is an issue where most of the API doesn't accept double precision
    double* mpHostCovariance = nullptr;
    double* mpHostCovarianceConditioning = nullptr;
    double* mpHostObservationsConditioning = nullptr;
    double* mpHostCovarianceCross = nullptr;
    double* mpHostLogDetResults = nullptr;
    double* mpHostNorm2Results = nullptr;
    
    // Device memory
    double* mpDeviceCovariance = nullptr;
    double* mpDeviceObservations = nullptr;
    double* mpDeviceObservationsCopy = nullptr;
    double* mpDeviceCovarianceConditioning = nullptr;
    double* mpDeviceObservationsConditioning = nullptr;
    double* mpDeviceObservationsConditioningCopy = nullptr;
    double* mpDeviceCovarianceCross = nullptr;
    double* mpDeviceCovarianceOffset = nullptr;
    double* mpDeviceMuOffset = nullptr;
    
    // Batch arrays (array of pointers)
    T** mpHostCovarianceArray = nullptr;
    T** mpDeviceCovarianceArray = nullptr;
    T** mpHostObservationsArray = nullptr;
    T** mpDeviceObservationsArray = nullptr;
    T** mpHostObservationsArrayCopy = nullptr;
    T** mpDeviceObservationsArrayCopy = nullptr;
    T** mpHostCovarianceConditioningArray = nullptr;
    T** mpDeviceCovarianceConditioningArray = nullptr;
    T** mpHostCovarianceCrossArray = nullptr;
    T** mpDeviceCovarianceCrossArray = nullptr;
    T** mpHostCovarianceOffsetArray = nullptr;
    T** mpDeviceCovarianceOffsetArray = nullptr;
    T** mpHostMuOffsetArray = nullptr;
    T** mpDeviceMuOffsetArray = nullptr;
    T** mpHostObservationsConditioningArray = nullptr;
    T** mpDeviceObservationsConditioningArray = nullptr;
    T** mpHostObservationsConditioningArrayCopy = nullptr;
    T** mpDeviceObservationsConditioningArrayCopy = nullptr;
    
    // Leading dimensions
    magma_int_t* mpHostLDA = nullptr;
    magma_int_t* mpHostLDDA = nullptr;
    magma_int_t* mpDeviceLDA = nullptr;
    magma_int_t* mpDeviceLDDA = nullptr;
    magma_int_t* mpHostLDAConditioning = nullptr;
    magma_int_t* mpHostLDDAConditioning = nullptr;
    magma_int_t* mpDeviceLDAConditioning = nullptr;
    magma_int_t* mpDeviceLDDAConditioning = nullptr;
    
    // Info and constants
    magma_int_t* mpHostInfo = nullptr;
    magma_int_t* mpDeviceInfo = nullptr;
    magma_int_t* mpHostConst1 = nullptr;
    magma_int_t* mpDeviceConst1 = nullptr;
    magma_int_t* mpDeviceBatchNum = nullptr;
    
    // Memory size tracking
    long long mTotalSizeCpuCovariance = 0;
    long long mTotalSizeDevCovariance = 0;
    long long mTotalSizeCpuObservations = 0;
    long long mTotalSizeDevObservations = 0;
    long long mTotalSizeDeviceObservations = 0;
    
    // ========== Multi-GPU arrays for Parallel Vecchia (matching llh_data struct) ==========
    int mNumGPUs = 0;
    int* mpBatchCountGPU = nullptr;  // batchCount_gpu[ngpu]
    
    // Host result arrays per GPU
    double** mpDotResultH = nullptr;      // dot_result_h[ngpu]
    double** mpLogdetResultH = nullptr;   // logdet_result_h[ngpu]
    
    // Device arrays per GPU
    double** mpDeviceC = nullptr;                    // d_C[ngpu]
    int** mpDeviceInfoArray = nullptr;               // d_info[ngpu]
    double** mpLocationsXXD = nullptr;               // locations_xx_d[ngpu]
    double** mpLocationsYYD = nullptr;               // locations_yy_d[ngpu]
    double** mpLocationsConXXD = nullptr;            // locations_con_xx_d[ngpu]
    double** mpLocationsConYYD = nullptr;            // locations_con_yy_d[ngpu]
    
    // Note: The following are already declared above as T** for Block Vecchia:
    // - mpDeviceAConditioningArray (as mpDeviceCovarianceConditioningArray)
    // - mpDeviceACrossArray (as mpDeviceCovarianceCrossArray)  
    // - mpDeviceAOffsetArray (as mpDeviceCovarianceOffsetArray)
    // - mpDeviceMuOffsetArray (already exists)
    
    //// Scaled Block Vecchia data
    std::vector<vecchia::dataunits::BlockInfo> mBlockInfos;
    
    //// Timing data for logging
    vecchia::utilities::TimingData mTimingData;
    
    //// Scaled Block Vecchia test/prediction data
    std::vector<vecchia::dataunits::BlockInfo> mBlockInfos_test;
    
    //// Clustering result for test/prediction locations (stored as pointer to avoid circular dependency)
    std::unique_ptr<vecchia::clustering::ClusteringResult<T>> mpTestClusteringResult;
};

/**
 * @brief Instantiates the VecchiaGB class for float and double types.
 * @tparam T Data Type: float or double
 */
VECCHIAGP_INSTANTIATE_CLASS(VecchiaGBData)

#endif //VECCHIAGP_DATA_HPP