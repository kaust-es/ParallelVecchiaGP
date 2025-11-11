
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file LocationSorter.cpp
 * @brief Implementation of location sorting and reordering methods.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-18
**/

#include <helpers/LocationSorter.hpp>
#include <utilities/Logger.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <tuple>
#include <cmath>

using namespace vecchia::helpers;
using namespace vecchia::configurations;
using namespace vecchia::common;
using namespace vecchia::dataunits;

template<typename T>
LocationSorter<T>& LocationSorter<T>::GetInstance() {
    static LocationSorter<T> instance;
    return instance;
}

template<typename T>
void LocationSorter<T>::ApplyReordering(const OrderingMethod &aOrderingMethod, const int &aN,
                                      const Dimension &aDimension, Locations<T> &aLocations, T *aObservations) {
    switch (aOrderingMethod) {
        case OrderingMethod::RANDOM:
            LOGGER("--------You are using Random Ordering---------");
            RandomReordering(aN, aDimension, aLocations, aObservations);
            break;
        case OrderingMethod::MORTON:
            LOGGER("--------You are using Morton Ordering---------");
            MortonReordering(aN, aDimension, aLocations, aObservations);
            break;
        case OrderingMethod::KD_TREE:
            LOGGER("--------You are using KD-Tree Ordering---------");
            KDTreeReordering(aN, aDimension, aLocations, aObservations);
            break;
        case OrderingMethod::HILBERT:
            LOGGER("--------You are using Hilbert Ordering---------");
            HilbertReordering(aN, aDimension, aLocations, aObservations);
            break;
        case OrderingMethod::MMD:
            LOGGER("--------You are using MMD Ordering---------");
            MMDReordering(aN, aDimension, aLocations, aObservations);
            break;
        default:
            LOGGER("--------Unknown ordering method, defaulting to Random---------");
            RandomReordering(aN, aDimension, aLocations, aObservations);
            break;
    }
}

template<typename T>
void LocationSorter<T>::RandomReordering(const int &aN, const Dimension &aDimension,
                                       Locations<T> &aLocations, T *aObservations) {
    int seed = 42;
    srand(seed);

    for (int i = aN - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap x values
        T tempX = aLocations.GetLocationX()[i];
        aLocations.GetLocationX()[i] = aLocations.GetLocationX()[j];
        aLocations.GetLocationX()[j] = tempX;

        // Swap y values
        T tempY = aLocations.GetLocationY()[i];
        aLocations.GetLocationY()[i] = aLocations.GetLocationY()[j];
        aLocations.GetLocationY()[j] = tempY;

        // Swap z values for 3D
        if (aDimension == Dimension3D) {
            T tempZ = aLocations.GetLocationZ()[i];
            aLocations.GetLocationZ()[i] = aLocations.GetLocationZ()[j];
            aLocations.GetLocationZ()[j] = tempZ;
        }
        
        // Swap observations if provided
        if (aObservations != nullptr) {
            T tempObs = aObservations[i];
            aObservations[i] = aObservations[j];
            aObservations[j] = tempObs;
        }
    }
}

// Morton encoding/decoding helper functions for 2D
template<typename T>
uint32_t LocationSorter<T>::Part1By1(uint32_t x) {
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

template<typename T>
uint32_t LocationSorter<T>::Compact1By1(uint32_t x) {
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

template<typename T>
uint32_t LocationSorter<T>::EncodeMorton2(uint32_t x, uint32_t y) {
    return (Part1By1(y) << 1) + Part1By1(x);
}

template<typename T>
uint32_t LocationSorter<T>::DecodeMorton2X(uint32_t code) {
    return Compact1By1(code >> 0);
}

template<typename T>
uint32_t LocationSorter<T>::DecodeMorton2Y(uint32_t code) {
    return Compact1By1(code >> 1);
}

// Morton encoding/decoding helper functions for 3D
template<typename T>
uint64_t LocationSorter<T>::Part1By3(uint64_t x) {
    x &= 0x000000000000ffff;
    x = (x ^ (x << 24)) & 0x000000ff000000ff;
    x = (x ^ (x << 12)) & 0x000f000f000f000f;
    x = (x ^ (x << 6)) & 0x0303030303030303;
    x = (x ^ (x << 3)) & 0x1111111111111111;
    return x;
}

template<typename T>
uint64_t LocationSorter<T>::Compact1By3(uint64_t x) {
    x &= 0x1111111111111111;
    x = (x ^ (x >> 3)) & 0x0303030303030303;
    x = (x ^ (x >> 6)) & 0x000f000f000f000f;
    x = (x ^ (x >> 12)) & 0x000000ff000000ff;
    x = (x ^ (x >> 24)) & 0x000000000000ffff;
    return x;
}

template<typename T>
uint64_t LocationSorter<T>::EncodeMorton3(uint64_t x, uint64_t y, uint64_t z) {
    return (Part1By3(z) << 2) + (Part1By3(y) << 1) + Part1By3(x);
}

template<typename T>
uint64_t LocationSorter<T>::DecodeMorton3X(uint64_t code) {
    return Compact1By3(code >> 0);
}

template<typename T>
uint64_t LocationSorter<T>::DecodeMorton3Y(uint64_t code) {
    return Compact1By3(code >> 1);
}

template<typename T>
uint64_t LocationSorter<T>::DecodeMorton3Z(uint64_t code) {
    return Compact1By3(code >> 2);
}

template<typename T>
void LocationSorter<T>::MortonReordering(const int &aN, const Dimension &aDimension,
                                        Locations<T> &aLocations, T *aObservations) {
    if (aDimension == Dimension2D || aDimension == DimensionST) {
        // 2D Morton ordering - use struct to keep track of observations
        struct DataPair {
            uint32_t code;
            T obs;
        };
        std::vector<DataPair> encoded(aN);
        
        for (int i = 0; i < aN; i++) {
            uint16_t x = (uint16_t)(aLocations.GetLocationX()[i] * (double)UINT16_MAX + 0.5);
            uint16_t y = (uint16_t)(aLocations.GetLocationY()[i] * (double)UINT16_MAX + 0.5);
            encoded[i].code = EncodeMorton2(x, y);
            if (aObservations != nullptr) {
                encoded[i].obs = aObservations[i];
            }
        }
        
        std::sort(encoded.begin(), encoded.end(), [](const DataPair& a, const DataPair& b) {
            return a.code < b.code;
        });
        
        // Create temporary arrays to store sorted values
        T* tempX = new T[aN];
        T* tempY = new T[aN];
        
        for (int i = 0; i < aN; i++) {
            uint16_t x = DecodeMorton2X(encoded[i].code);
            uint16_t y = DecodeMorton2Y(encoded[i].code);
            tempX[i] = (T)x / (T)UINT16_MAX;
            tempY[i] = (T)y / (T)UINT16_MAX;
            if (aObservations != nullptr) {
                aObservations[i] = encoded[i].obs;
            }
        }
        
        // Copy back to locations
        std::memcpy(aLocations.GetLocationX(), tempX, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationY(), tempY, aN * sizeof(T));
        
        delete[] tempX;
        delete[] tempY;
    } else {
        // 3D Morton ordering
        struct DataPair {
            uint64_t code;
            T obs;
        };
        std::vector<DataPair> encoded(aN);
        
        for (int i = 0; i < aN; i++) {
            uint16_t x = (uint16_t)(aLocations.GetLocationX()[i] * (double)UINT16_MAX + 0.5);
            uint16_t y = (uint16_t)(aLocations.GetLocationY()[i] * (double)UINT16_MAX + 0.5);
            uint16_t z = (uint16_t)(aLocations.GetLocationZ()[i] * (double)UINT16_MAX + 0.5);
            encoded[i].code = EncodeMorton3(x, y, z);
            if (aObservations != nullptr) {
                encoded[i].obs = aObservations[i];
            }
        }
        
        std::sort(encoded.begin(), encoded.end(), [](const DataPair& a, const DataPair& b) {
            return a.code < b.code;
        });
        
        // Create temporary arrays
        T* tempX = new T[aN];
        T* tempY = new T[aN];
        T* tempZ = new T[aN];
        
        for (int i = 0; i < aN; i++) {
            uint16_t x = DecodeMorton3X(encoded[i].code);
            uint16_t y = DecodeMorton3Y(encoded[i].code);
            uint16_t z = DecodeMorton3Z(encoded[i].code);
            tempX[i] = (T)x / (T)UINT16_MAX;
            tempY[i] = (T)y / (T)UINT16_MAX;
            tempZ[i] = (T)z / (T)UINT16_MAX;
            if (aObservations != nullptr) {
                aObservations[i] = encoded[i].obs;
            }
        }
        
        std::memcpy(aLocations.GetLocationX(), tempX, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationY(), tempY, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationZ(), tempZ, aN * sizeof(T));
        
        delete[] tempX;
        delete[] tempY;
        delete[] tempZ;
    }
}

// Hilbert curve helper functions
template<typename T>
uint32_t LocationSorter<T>::EncodeHilbert2(uint32_t x, uint32_t y) {
    uint32_t M = 1 << 15, P, Q, t;
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        if (x & Q)
            x ^= P;
        else {
            t = (x ^ x) & P;
            x ^= t;
            x ^= t;
        }
        if (y & Q)
            x ^= P;
        else {
            t = (x ^ y) & P;
            x ^= t;
            y ^= t;
        }
    }
    y ^= x;
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
        if (y & Q)
            t ^= Q - 1;
    x ^= t;
    y ^= t;
    uint32_t result = 0;
    uint32_t res;
    for (int i = 0; i < 16; i++) {
        res = x >> (15 - i) & 1;
        result |= res << (31 - i * 2);
        res = y >> (15 - i) & 1;
        result |= res << (31 - (i * 2 + 1));
    }
    return result;
}

template<typename T>
void LocationSorter<T>::DecodeHilbert2(uint32_t result, uint32_t &x, uint32_t &y) {
    uint32_t N = 2 << 15, P, Q, t;
    x = 0;
    y = 0;
    uint32_t res;
    for (int i = 0; i < 16; i++) {
        res = result >> 31;
        result = result << 1;
        x |= res << (15 - i);
        res = result >> 31;
        result = result << 1;
        y |= res << (15 - i);
    }
    t = y >> 1;
    y ^= x;
    x ^= t;
    for (Q = 2; Q != N; Q <<= 1) {
        P = Q - 1;
        if (y & Q)
            x ^= P;
        else {
            t = (x ^ y) & P;
            x ^= t;
            y ^= t;
        }
        if (x & Q)
            x ^= P;
        else {
            t = (x ^ x) & P;
            x ^= t;
            x ^= t;
        }
    }
}

template<typename T>
uint64_t LocationSorter<T>::EncodeHilbert3(uint64_t x, uint64_t y, uint64_t z) {
    uint16_t M = 1 << 15, P, Q, t;
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        if (x & Q)
            x ^= P;
        else {
            t = (x ^ x) & P;
            x ^= t;
            x ^= t;
        }
        if (y & Q)
            x ^= P;
        else {
            t = (x ^ y) & P;
            x ^= t;
            y ^= t;
        }
        if (z & Q)
            x ^= P;
        else {
            t = (x ^ z) & P;
            x ^= t;
            z ^= t;
        }
    }
    y ^= x;
    z ^= y;
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
        if (z & Q)
            t ^= Q - 1;
    x ^= t;
    y ^= t;
    z ^= t;
    uint64_t result = 0;
    uint64_t res;
    for (int i = 0; i < 16; i++) {
        res = x >> (15 - i) & 1;
        result |= res << (47 - i * 3);
        res = y >> (15 - i) & 1;
        result |= res << (47 - (i * 3 + 1));
        res = z >> (15 - i) & 1;
        result |= res << (47 - (i * 3 + 2));
    }
    return result;
}

template<typename T>
void LocationSorter<T>::DecodeHilbert3(uint64_t result, uint64_t &x, uint64_t &y, uint64_t &z) {
    uint64_t N = 2 << 15, P, Q, t;
    x = 0;
    y = 0;
    z = 0;
    uint64_t res;
    result = result << 16;
    for (int i = 0; i < 16; i++) {
        res = result >> 63 & 1;
        result = result << 1;
        x |= res << (15 - i);
        res = result >> 63 & 1;
        result = result << 1;
        y |= res << (15 - i);
        res = result >> 63 & 1;
        result = result << 1;
        z |= res << (15 - i);
    }
    t = z >> 1;
    z ^= y;
    y ^= x;
    x ^= t;
    for (Q = 2; Q != N; Q <<= 1) {
        P = Q - 1;
        if (z & Q)
            x ^= P;
        else {
            t = (x ^ z) & P;
            x ^= t;
            z ^= t;
        }
        if (y & Q)
            x ^= P;
        else {
            t = (x ^ y) & P;
            x ^= t;
            y ^= t;
        }
        if (x & Q)
            x ^= P;
        else {
            t = 0 & P;
            x ^= t;
            x ^= t;
        }
    }
}

template<typename T>
void LocationSorter<T>::HilbertReordering(const int &aN, const Dimension &aDimension,
                                         Locations<T> &aLocations, T *aObservations) {
    if (aDimension == Dimension2D || aDimension == DimensionST) {
        // 2D Hilbert ordering
        std::vector<std::pair<uint32_t, int>> encoded(aN);
        
        for (int i = 0; i < aN; i++) {
            uint16_t x = (uint16_t)(aLocations.GetLocationX()[i] * (double)UINT16_MAX + 0.5);
            uint16_t y = (uint16_t)(aLocations.GetLocationY()[i] * (double)UINT16_MAX + 0.5);
            encoded[i] = {EncodeHilbert2(x, y), i};
        }
        
        std::sort(encoded.begin(), encoded.end());
        
        T* tempX = new T[aN];
        T* tempY = new T[aN];
        
        for (int i = 0; i < aN; i++) {
            uint32_t x, y;
            DecodeHilbert2(encoded[i].first, x, y);
            tempX[i] = (T)x / (T)UINT16_MAX;
            tempY[i] = (T)y / (T)UINT16_MAX;
        }
        
        std::memcpy(aLocations.GetLocationX(), tempX, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationY(), tempY, aN * sizeof(T));
        
        delete[] tempX;
        delete[] tempY;
    } else {
        // 3D Hilbert ordering
        std::vector<std::pair<uint64_t, int>> encoded(aN);
        
        for (int i = 0; i < aN; i++) {
            uint16_t x = (uint16_t)(aLocations.GetLocationX()[i] * (double)UINT16_MAX + 0.5);
            uint16_t y = (uint16_t)(aLocations.GetLocationY()[i] * (double)UINT16_MAX + 0.5);
            uint16_t z = (uint16_t)(aLocations.GetLocationZ()[i] * (double)UINT16_MAX + 0.5);
            encoded[i] = {EncodeHilbert3(x, y, z), i};
        }
        
        std::sort(encoded.begin(), encoded.end());
        
        T* tempX = new T[aN];
        T* tempY = new T[aN];
        T* tempZ = new T[aN];
        
        for (int i = 0; i < aN; i++) {
            uint64_t x, y, z;
            DecodeHilbert3(encoded[i].first, x, y, z);
            tempX[i] = (T)x / (T)UINT16_MAX;
            tempY[i] = (T)y / (T)UINT16_MAX;
            tempZ[i] = (T)z / (T)UINT16_MAX;
        }
        
        std::memcpy(aLocations.GetLocationX(), tempX, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationY(), tempY, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationZ(), tempZ, aN * sizeof(T));
        
        delete[] tempX;
        delete[] tempY;
        delete[] tempZ;
    }
}

// KD-Tree implementation
template<typename T>
typename LocationSorter<T>::TreeNode2D* LocationSorter<T>::BuildKDTree2D(
    std::vector<std::tuple<T, T>>& data, int depth) {
    
    if (data.empty())
        return nullptr;
    
    if (data.size() == 1) {
        TreeNode2D* leaf = new TreeNode2D();
        leaf->dim = -1;
        leaf->x = std::get<0>(data[0]);
        leaf->y = std::get<1>(data[0]);
        leaf->left = nullptr;
        leaf->right = nullptr;
        return leaf;
    }
    
    TreeNode2D* node = new TreeNode2D();
    int dim = depth % 2;
    node->dim = dim;
    
    int mid = data.size() / 2;
    
    if (dim == 0) {
        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        node->x = std::get<0>(data[mid]);
    } else {
        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
            return std::get<1>(a) < std::get<1>(b);
        });
        node->y = std::get<1>(data[mid]);
    }
    
    std::vector<std::tuple<T, T>> leftData(data.begin(), data.begin() + mid);
    std::vector<std::tuple<T, T>> rightData(data.begin() + mid, data.end());
    
    node->left = BuildKDTree2D(leftData, depth + 1);
    node->right = BuildKDTree2D(rightData, depth + 1);
    
    return node;
}

template<typename T>
void LocationSorter<T>::TraverseKDTree2D(TreeNode2D* root, std::vector<std::tuple<T, T>>& result) {
    if (root == nullptr)
        return;
    
    if (root->left != nullptr)
        TraverseKDTree2D(root->left, result);
    
    if (root->dim == -1) {
        result.push_back(std::make_tuple(root->x, root->y));
    }
    
    if (root->right != nullptr)
        TraverseKDTree2D(root->right, result);
}

template<typename T>
void LocationSorter<T>::FreeKDTree2D(TreeNode2D* root) {
    if (root == nullptr)
        return;
    
    FreeKDTree2D(root->left);
    FreeKDTree2D(root->right);
    delete root;
}

template<typename T>
typename LocationSorter<T>::TreeNode3D* LocationSorter<T>::BuildKDTree3D(
    std::vector<std::tuple<T, T, T>>& data, int depth) {
    
    if (data.empty())
        return nullptr;
    
    if (data.size() == 1) {
        TreeNode3D* leaf = new TreeNode3D();
        leaf->dim = -1;
        leaf->x = std::get<0>(data[0]);
        leaf->y = std::get<1>(data[0]);
        leaf->z = std::get<2>(data[0]);
        leaf->left = nullptr;
        leaf->right = nullptr;
        return leaf;
    }
    
    TreeNode3D* node = new TreeNode3D();
    int dim = depth % 3;
    node->dim = dim;
    
    int mid = data.size() / 2;
    
    if (dim == 0) {
        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        node->x = std::get<0>(data[mid]);
    } else if (dim == 1) {
        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
            return std::get<1>(a) < std::get<1>(b);
        });
        node->y = std::get<1>(data[mid]);
    } else {
        std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
            return std::get<2>(a) < std::get<2>(b);
        });
        node->z = std::get<2>(data[mid]);
    }
    
    std::vector<std::tuple<T, T, T>> leftData(data.begin(), data.begin() + mid);
    std::vector<std::tuple<T, T, T>> rightData(data.begin() + mid, data.end());
    
    node->left = BuildKDTree3D(leftData, depth + 1);
    node->right = BuildKDTree3D(rightData, depth + 1);
    
    return node;
}

template<typename T>
void LocationSorter<T>::TraverseKDTree3D(TreeNode3D* root, std::vector<std::tuple<T, T, T>>& result) {
    if (root == nullptr)
        return;
    
    if (root->left != nullptr)
        TraverseKDTree3D(root->left, result);
    
    if (root->dim == -1) {
        result.push_back(std::make_tuple(root->x, root->y, root->z));
    }
    
    if (root->right != nullptr)
        TraverseKDTree3D(root->right, result);
}

template<typename T>
void LocationSorter<T>::FreeKDTree3D(TreeNode3D* root) {
    if (root == nullptr)
        return;
    
    FreeKDTree3D(root->left);
    FreeKDTree3D(root->right);
    delete root;
}

template<typename T>
void LocationSorter<T>::KDTreeReordering(const int &aN, const Dimension &aDimension,
                                        Locations<T> &aLocations, T *aObservations) {
    if (aDimension == Dimension2D || aDimension == DimensionST) {
        // 2D KD-Tree
        std::vector<std::tuple<T, T>> data(aN);
        
        for (int i = 0; i < aN; i++) {
            data[i] = std::make_tuple(aLocations.GetLocationX()[i], aLocations.GetLocationY()[i]);
        }
        
        TreeNode2D* root = BuildKDTree2D(data, 0);
        
        std::vector<std::tuple<T, T>> result;
        TraverseKDTree2D(root, result);
        
        for (int i = 0; i < aN; i++) {
            aLocations.GetLocationX()[i] = std::get<0>(result[i]);
            aLocations.GetLocationY()[i] = std::get<1>(result[i]);
        }
        
        FreeKDTree2D(root);
    } else {
        // 3D KD-Tree
        std::vector<std::tuple<T, T, T>> data(aN);
        
        for (int i = 0; i < aN; i++) {
            data[i] = std::make_tuple(aLocations.GetLocationX()[i], 
                                     aLocations.GetLocationY()[i],
                                     aLocations.GetLocationZ()[i]);
        }
        
        TreeNode3D* root = BuildKDTree3D(data, 0);
        
        std::vector<std::tuple<T, T, T>> result;
        TraverseKDTree3D(root, result);
        
        for (int i = 0; i < aN; i++) {
            aLocations.GetLocationX()[i] = std::get<0>(result[i]);
            aLocations.GetLocationY()[i] = std::get<1>(result[i]);
            aLocations.GetLocationZ()[i] = std::get<2>(result[i]);
        }
        
        FreeKDTree3D(root);
    }
}

// MMD (Maximum Minimum Distance) implementation
template<typename T>
void LocationSorter<T>::MMDReordering(const int &aN, const Dimension &aDimension,
                                     Locations<T> &aLocations, T *aObservations) {
    if (aDimension == Dimension2D || aDimension == DimensionST) {
        // 2D MMD
        std::vector<int> res(aN);
        std::vector<bool> flag(aN, false);
        
        // Calculate mean
        T x_mean = 0, y_mean = 0;
        for (int i = 0; i < aN; i++) {
            x_mean += aLocations.GetLocationX()[i];
            y_mean += aLocations.GetLocationY()[i];
        }
        x_mean /= aN;
        y_mean /= aN;
        
        // Find point closest to mean
        T mindist = 2;
        for (int i = 0; i < aN; i++) {
            T dist = std::pow(aLocations.GetLocationX()[i] - x_mean, 2) + 
                     std::pow(aLocations.GetLocationY()[i] - y_mean, 2);
            if (dist < mindist) {
                mindist = dist;
                res[0] = i;
            }
        }
        flag[res[0]] = true;
        
        // Build ordering based on maximum minimum distance
        for (int j = 1; j < aN - 1; j++) {
            std::vector<T> max_list(aN, 0);
            
            for (int i = 0; i < aN; i++) {
                if (!flag[i]) {
                    T min_temp = 2;
                    for (int k = 0; k < j; k++) {
                        T temp = std::pow(aLocations.GetLocationX()[i] - aLocations.GetLocationX()[res[k]], 2) + 
                                 std::pow(aLocations.GetLocationY()[i] - aLocations.GetLocationY()[res[k]], 2);
                        if (temp < min_temp)
                            min_temp = temp;
                    }
                    max_list[i] = min_temp;
                }
            }
            
            T max_temp = 0;
            int ind_temp = aN;
            for (int i = 0; i < aN; i++) {
                if (max_temp < max_list[i]) {
                    max_temp = max_list[i];
                    ind_temp = i;
                }
            }
            res[j] = ind_temp;
            flag[res[j]] = true;
        }
        
        // Find last unflagged point
        for (int i = 0; i < aN; i++) {
            if (!flag[i])
                res[aN - 1] = i;
        }
        
        // Reorder locations
        T* tempX = new T[aN];
        T* tempY = new T[aN];
        
        for (int i = 0; i < aN; i++) {
            tempX[i] = aLocations.GetLocationX()[res[i]];
            tempY[i] = aLocations.GetLocationY()[res[i]];
        }
        
        std::memcpy(aLocations.GetLocationX(), tempX, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationY(), tempY, aN * sizeof(T));
        
        delete[] tempX;
        delete[] tempY;
    } else {
        // 3D MMD
        std::vector<int> res(aN);
        std::vector<bool> flag(aN, false);
        
        // Calculate mean
        T x_mean = 0, y_mean = 0, z_mean = 0;
        for (int i = 0; i < aN; i++) {
            x_mean += aLocations.GetLocationX()[i];
            y_mean += aLocations.GetLocationY()[i];
            z_mean += aLocations.GetLocationZ()[i];
        }
        x_mean /= aN;
        y_mean /= aN;
        z_mean /= aN;
        
        // Find point closest to mean
        T mindist = 3;
        for (int i = 0; i < aN; i++) {
            T dist = std::pow(aLocations.GetLocationX()[i] - x_mean, 2) + 
                     std::pow(aLocations.GetLocationY()[i] - y_mean, 2) +
                     std::pow(aLocations.GetLocationZ()[i] - z_mean, 2);
            if (dist < mindist) {
                mindist = dist;
                res[0] = i;
            }
        }
        flag[res[0]] = true;
        
        // Build ordering
        for (int j = 1; j < aN - 1; j++) {
            std::vector<T> max_list(aN, 0);
            
            for (int i = 0; i < aN; i++) {
                if (!flag[i]) {
                    T min_temp = 3;
                    for (int k = 0; k < j; k++) {
                        T temp = std::pow(aLocations.GetLocationX()[i] - aLocations.GetLocationX()[res[k]], 2) + 
                                 std::pow(aLocations.GetLocationY()[i] - aLocations.GetLocationY()[res[k]], 2) +
                                 std::pow(aLocations.GetLocationZ()[i] - aLocations.GetLocationZ()[res[k]], 2);
                        if (temp < min_temp)
                            min_temp = temp;
                    }
                    max_list[i] = min_temp;
                }
            }
            
            T max_temp = 0;
            int ind_temp = aN;
            for (int i = 0; i < aN; i++) {
                if (max_temp < max_list[i]) {
                    max_temp = max_list[i];
                    ind_temp = i;
                }
            }
            res[j] = ind_temp;
            flag[res[j]] = true;
        }
        
        // Find last unflagged point
        for (int i = 0; i < aN; i++) {
            if (!flag[i])
                res[aN - 1] = i;
        }
        
        // Reorder locations
        T* tempX = new T[aN];
        T* tempY = new T[aN];
        T* tempZ = new T[aN];
        
        for (int i = 0; i < aN; i++) {
            tempX[i] = aLocations.GetLocationX()[res[i]];
            tempY[i] = aLocations.GetLocationY()[res[i]];
            tempZ[i] = aLocations.GetLocationZ()[res[i]];
        }
        
        std::memcpy(aLocations.GetLocationX(), tempX, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationY(), tempY, aN * sizeof(T));
        std::memcpy(aLocations.GetLocationZ(), tempZ, aN * sizeof(T));
        
        delete[] tempX;
        delete[] tempY;
        delete[] tempZ;
    }
}

