/**
 * @file ModelingDataHolders.hpp
 * @brief This file contains the definition of the mModelingData struct, which contains all the data needed for modeling.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_MODELINGDATAHOLDERS_HPP
#define VECCHIAGP_MODELINGDATAHOLDERS_HPP

namespace vecchia::dataunits {

    /**
     * @brief Struct containing all the data needed for modeling.
     * @tparam T The data type of the data.
     */
    template<typename T>
    struct mModelingData {
        /// VecchiaGBData<T> object containing needed descriptors, and locations.
        std::unique_ptr<VecchiaGBData<T>> *mpData;
        /// Configurations object containing user input data.
        configurations::Configurations *mpConfiguration;
        /// Used Kernel for VecchiaGB Modeling Data.
        const kernels::Kernel<T> *mpKernel;

        /// User Input Measurements Matrix
        T *mpMeasurementsMatrix;

        /**
         * @brief Constructor.
         * @param aData The VecchiaGBData object.
         * @param aConfiguration The Configurations object.
         * @param aKernel The Kernel object.
         */
        mModelingData(std::unique_ptr<VecchiaGBData<T>> &aData, configurations::Configurations &aConfiguration,
                      T &aMatrix, const kernels::Kernel<T> &aKernel) :
                mpData(std::move(&aData)), mpConfiguration(&aConfiguration), mpMeasurementsMatrix(&aMatrix),
                mpKernel(&aKernel) {}
    };

}//namespace vecchia
#endif //VECCHIAGP_MODELINGDATAHOLDERS_HPP