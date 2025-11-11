
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file EnumStringParser.hpp
 * @brief Provides utility functions for parsing enumeration values from strings.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2024-01-20
**/

#ifndef VECCHIAGP_ENUMSTRINGPARSER_HPP
#define VECCHIAGP_ENUMSTRINGPARSER_HPP

#include <algorithm>

#include <utilities/ErrorHandler.hpp>
#include <common/Definitions.hpp>


/**
 * @brief Converts string to dimension enum.
 * @param[in] aDimension Dimension as a string.
 * @return Dimension as an enum.
 */
inline vecchia::common::Dimension GetInputDimension(std::string aDimension) {
    std::transform(aDimension.begin(), aDimension.end(),
                   aDimension.begin(), ::tolower);

    if (aDimension == "2d") {
        return vecchia::common::Dimension2D;
    } else if (aDimension == "3d") {
        return vecchia::common::Dimension3D;
    } else if (aDimension == "st") {
        return vecchia::common::DimensionST;
    } else {
        const std::string msg = "Error in Initialization : Unknown computation Value" + std::string(aDimension);
        throw API_EXCEPTION(msg, INVALID_ARGUMENT_ERROR);
    }
}

#endif //VECCHIAGP_ENUMSTRINGPARSER_HPP
