
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file  CSVLoader.hpp
 * @brief A class for generating synthetic data.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGBCPP_CSVDATALOADER_HPP
#define VECCHIAGBCPP_CSVDATALOADER_HPP

#include <data-loader/DataLoader.hpp>

namespace vecchia::dataLoader::csv {

    /**
     * @class  CSVLoader
     * @brief A class for creating data by reading CSV files.
     * @tparam T Data Type: float or double
     */
    template<typename T>
    class CSVLoader : public DataLoader<T> {
    public:

        /**
         * @brief Get a pointer to the singleton instance of the  CSVLoader class.
         * @return A pointer to the instance of the  CSVLoader class.
         *
         */
        static CSVLoader<T> *GetInstance();

        /**
         * @brief Reads data from external sources into VecchiaGB format.
         * @copydoc DataLoader::ReadData()
         *
         */
        void ReadData(configurations::Configurations &aConfigurations, std::vector<T> &aMeasurementsMatrix,
                      std::vector<T> &aXLocations, std::vector<T> &aYLocations, std::vector<T> &aZLocations,
                      const int &aP) override;

        /**
        * @brief Writes a matrix of vectors to disk.
        * @copydoc DataLoader::WriteData()
        *
        */
        void
        WriteData(const T &aMatrixPointer, const int &aProblemSize, const int &aP, std::string &aLoggerPath,
                  vecchia::dataunits::Locations<T> &aLocations) override;

        /**
         * @brief Release the singleton instance of the  CSVLoader class.
         * @return void
         *
         */
        static void ReleaseInstance();

    private:
        /**
         * @brief Constructor for the  CSVLoader class.
         * @return void
         *
         */
        CSVLoader() = default;

        /**
         * @brief Default destructor.
         *
         */
        ~CSVLoader() override = default;

        /**
         * @brief Pointer to the singleton instance of the  CSVLoader class.
         *
         */
        static CSVLoader<T> *mpInstance;

    };

    /**
     * @brief Instantiates the CSV Data Generator class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(CSVLoader)
}
#endif //VECCHIAGBCPP_CSVDATALOADER_HPP