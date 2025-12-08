/**
 *
 * Copyright (c) 2017-2023  King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * ExaGeoStat is a software package provided by KAUST
 **/
/**
 *
 * @file ckernel.h
 *
 * Core functions header file.
 *
 * @version 1.2.0
 *
 * @author Sameh Abdulah
 * @date 2022-11-09
 *
 **/

#ifndef _CKERNEL_H_
#define _CKERNEL_H_
#include <stdint.h>
#include <stdlib.h>
// #define assert(ignore)((void) 0)
#include <cstring>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        double *x; ///< Values in X dimension.
        double *y; ///< Values in Y dimension.
        double *z; ///< Values in Z dimension.
    } location;

    location *GenerateXYLoc(int n, int seed);
    location *GenerateXYLoc_ST(int n, int t_slots, int seed);

    static uint32_t Part1By1(uint32_t x)
    //! Spread lower bits of input
    {
        x &= 0x0000ffff;
        // x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x << 8)) & 0x00ff00ff;
        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x << 4)) & 0x0f0f0f0f;
        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x << 2)) & 0x33333333;
        // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x << 1)) & 0x55555555;
        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        return x;
    }
    static uint32_t Compact1By1(uint32_t x)
    //! Collect every second bit into lower part of input
    {
        x &= 0x55555555;
        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x = (x ^ (x >> 1)) & 0x33333333;
        // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x >> 2)) & 0x0f0f0f0f;
        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x >> 4)) & 0x00ff00ff;
        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x >> 8)) & 0x0000ffff;
        // x = ---- ---- ---- ---- fedc ba98 7654 3210
        return x;
    }

    static uint32_t EncodeMorton2(uint32_t x, uint32_t y)
    //! Encode two inputs into one
    {
        return (Part1By1(y) << 1) + Part1By1(x);
    }

    static uint32_t DecodeMorton2X(uint32_t code)
    //! Decode first input
    {
        return Compact1By1(code >> 0);
    }

    static uint32_t DecodeMorton2Y(uint32_t code)
    //! Decode second input
    {
        return Compact1By1(code >> 1);
    }

    double uniform_distribution(double rangeLow, double rangeHigh);

    static int compare_uint32(const void *a, const void *b)
    //! Compare two uint32_t
    {
        uint32_t _a = *(uint32_t *)a;
        uint32_t _b = *(uint32_t *)b;
        if (_a < _b)
            return -1;
        if (_a == _b)
            return 0;
        return 1;
    }

    static void zsort_locations(int n, location *locations)
    //! Sort in Morton order (input points must be in [0;1]x[0;1] square])
    {
        // Some sorting, required by spatial statistics code
        int i;
        uint16_t x, y;
        uint32_t z[n];
        // Encode data into vector z
        for (i = 0; i < n; i++)
        {
            x = (uint16_t)(locations->x[i] * (double)UINT16_MAX + .5);
            y = (uint16_t)(locations->y[i] * (double)UINT16_MAX + .5);
            z[i] = EncodeMorton2(x, y);
        }
        // Sort vector z
        qsort(z, n, sizeof(uint32_t), compare_uint32);
        // Decode data from vector z
        for (i = 0; i < n; i++)
        {
            x = DecodeMorton2X(z[i]);
            y = DecodeMorton2Y(z[i]);
            locations->x[i] = (double)x / (double)UINT16_MAX;
            locations->y[i] = (double)y / (double)UINT16_MAX;
        }
    }
    struct comb
    { // location and measure
        uint32_t z;
        double w;
    };

    static int cmpfunc_loc(const void *a, const void *b)
    {
        struct comb _a = *(const struct comb *)a;
        struct comb _b = *(const struct comb *)b;
        if (_a.z < _b.z)
            return -1;
        if (_a.z == _b.z)
            return 0;
        return 1;
    }

    static void zsort_reordering(int n, location *locations)
    //! Sort in Morton order (input points must be in [0;1]x[0;1] square])
    {
        int i;
        int n_measurement = 1;
        uint16_t x, y;
        struct comb *dat = (struct comb *)malloc(n * n_measurement * sizeof(struct comb));
        // Encode data into vector z
        for (i = 0; i < n; i++)
        {
            x = (uint16_t)(locations->x[i] * (double)UINT16_MAX + .5);
            y = (uint16_t)(locations->y[i] * (double)UINT16_MAX + .5);
            dat[i].z = EncodeMorton2(x, y);
            // dat[i].w = w[i];
        }
        // Sort vector z
        qsort(dat, n, sizeof(struct comb), cmpfunc_loc);
        // Decode data from vector z
        for (i = 0; i < n; i++)
        {
            x = DecodeMorton2X(dat[i].z);
            y = DecodeMorton2Y(dat[i].z);
            locations->x[i] = (double)x / (double)UINT16_MAX;
            locations->y[i] = (double)y / (double)UINT16_MAX;
            // w[i] = dat[i].w;
        }
    }

    // random ordering
    static void random_reordering(int size, location *loc)
    {
        int your_seed_value = 42; // Set your desired seed value

        srand(your_seed_value);

        for (int i = size - 1; i > 0; i--)
        {
            int j = rand() % (i + 1);

            // Swap x values
            double tempX = loc->x[i];
            loc->x[i] = loc->x[j];
            loc->x[j] = tempX;

            // Swap y values
            double tempY = loc->y[i];
            loc->y[i] = loc->y[j];
            loc->y[j] = tempY;

            // // Swap obs values
            // double tempObs = h_C[i];
            // h_C[i] = h_C[j];
            // h_C[j] = tempObs;
        }
    }

    // kd tree
    struct tree
    {
        int dim;
        double x, y;
        double w;
        struct tree *left, *right;
    };

    struct arr
    {
        double x, y;
        double w;
    };

    static int cmpfunc_x(const void *a, const void *b)
    {
        struct arr _a = *(const struct arr *)a;
        struct arr _b = *(const struct arr *)b;
        if (_a.x < _b.x)
            return -1;
        if (_a.x == _b.x)
            return 0;
        return 1;
    }

    static int cmpfunc_y(const void *a, const void *b)
    {
        struct arr _a = *(const struct arr *)a;
        struct arr _b = *(const struct arr *)b;
        if (_a.y < _b.y)
            return -1;
        if (_a.y == _b.y)
            return 0;
        return 1;
    }

    static struct tree *initialize(int len, struct arr *dat, int depth)
    {
        if (len < 1)
            return NULL;
        if (len == 1)
        {
            struct tree *root = (struct tree *)malloc(sizeof(struct tree));
            root->dim = -1;
            root->x = dat[0].x;
            root->y = dat[0].y;
            root->w = dat[0].w;
            root->left = NULL;
            root->right = NULL;
            return root;
        }
        struct tree *root = (struct tree *)malloc(sizeof(struct tree));
        // int dim = (x[end] - x[start] > y[end] - y[start]) ? 1 : 2;
        int dim = depth % 2;
        root->dim = dim;
        int mid = len / 2;
        if (dim == 0)
            qsort(dat, len, sizeof(struct arr), cmpfunc_x);
        else
            qsort(dat, len, sizeof(struct arr), cmpfunc_y);
        // root->len = len;
        if (dim == 0)
            root->x = dat[mid].x;
        else if (dim == 1)
            root->y = dat[mid].y;
        struct arr *ldat = (struct arr *)malloc(mid * sizeof(struct arr));
        struct arr *rdat = (struct arr *)malloc((len - mid) * sizeof(struct arr));
        memcpy(ldat, dat, mid * sizeof(struct arr));
        memcpy(rdat, dat + mid, (len - mid) * sizeof(struct arr));
        free(dat);
        root->left = initialize(mid, ldat, depth + 1);
        root->right = initialize(len - mid, rdat, depth + 1);
        // root->left_l = root->right_l = NULL;
        return root;
    }

    static int traverse(struct tree *root, int i, double *x, double *y, double *w)
    {
        if (root->left != NULL)
            i = traverse(root->left, i, x, y, w);
        if (root->dim == -1)
        {
            // printf("%d\n", i);
            x[i] = root->x;
            y[i] = root->y;
            w[i++] = root->w;
        }
        if (root->right != NULL)
            i = traverse(root->right, i, x, y, w);
        return i;
    }

    static void zsort_locations_kdtree(int n, location *locations)
    //! Sort in KD-Tree order (input points must be in [0;1]x[0;1] square])
    {
        // Some sorting, required by spatial statistics code
        int i;
        double *w = (double *)malloc(n * sizeof(double));
        struct arr *dat = (struct arr *)malloc(n * sizeof(struct arr));
        double x[n], y[n];

        // Encode data into vector z

        for (i = 0; i < n; i++)
        {
            dat[i].x = locations->x[i];
            dat[i].y = locations->y[i];
            dat[i].w = w[i];
        }
        struct tree *root = initialize(n, dat, 0);
        int index = traverse(root, 0, x, y, w);
        // printf("traversed\n");

        for (i = 0; i < n; i++)
        {
            locations->x[i] = x[i];
            locations->y[i] = y[i];
        }
        free(w);
    }

    // hilbert ordering
    static uint32_t EncodeHilbert2(uint32_t x, uint32_t y)
    {
        uint32_t M = 1 << 15, P, Q, t;
        for (Q = M; Q > 1; Q >>= 1)
        {
            P = Q - 1;
            if (x & Q)
                x ^= P;
            else
            {
                t = (x ^ x) & P;
                x ^= t;
                x ^= t;
            }
            if (y & Q)
                x ^= P;
            else
            {
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
        for (int i = 0; i < 16; i++)
        {
            res = x >> (15 - i) & 1;
            result |= res << (31 - i * 2);
            res = y >> (15 - i) & 1;
            result |= res << (31 - (i * 2 + 1));
        }
        return result;
    }

    static uint32_t *DecodeHilbert2(uint32_t result)
    {
        uint32_t N = 2 << 15, P, Q, t;
        uint32_t x = 0, y = 0;
        static uint32_t X[2] = {0, 0};
        uint32_t res;
        for (int i = 0; i < 16; i++)
        {
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
        for (Q = 2; Q != N; Q <<= 1)
        {
            P = Q - 1;
            if (y & Q)
                x ^= P;
            else
            {
                t = (x ^ y) & P;
                x ^= t;
                y ^= t;
            }
            if (x & Q)
                x ^= P;
            else
            {
                t = (x ^ x) & P;
                x ^= t;
                x ^= t;
            }
        }
        X[0] = x;
        X[1] = y;
        return X;
    }

    static void zsort_locations_hilbert(int n, location *locations)
    //! Sort in Hilbert order (input points must be in [0;1]x[0;1] square])
    {
        // Some sorting, required by spatial statistics code
        int i;
        uint16_t x, y;
        // uint32_t z[n]; //, z2[n];
        double *w = (double *)malloc(n * sizeof(double));
        int n_measurement = 1;
        struct comb *dat = (struct comb *)malloc(n * n_measurement * sizeof(struct comb));

        for (i = 0; i < n; i++)
        {
            x = (uint16_t)(locations->x[i] * (double)UINT16_MAX + .5);
            y = (uint16_t)(locations->y[i] * (double)UINT16_MAX + .5);
            dat[i].z = EncodeHilbert2(x, y);
            dat[i].w = w[i];
        }
        // Sort vector z
        qsort(dat, n, sizeof(struct comb), cmpfunc_loc);
        // Decode data from vector z
        for (i = 0; i < n; i++)
        {
            uint32_t *X = DecodeHilbert2(dat[i].z);
            x = *X;
            y = *(X + 1);
            locations->x[i] = (double)x / (double)UINT16_MAX;
            locations->y[i] = (double)y / (double)UINT16_MAX;
            w[i] = dat[i].w;
        }
        free(w);
    }

    // mmd
    static void zsort_locations_mmd(int n, location *locations)
    //! Sort in MMD order (input points must be in [0;1]x[0;1] square])
    {
        int res[n], flag[n];
        double *w = (double *)malloc(n * sizeof(double));
        struct arr *dat = (struct arr *)malloc(n * sizeof(struct arr));
        int n_measurement = 1;

        for (int i = 0; i < n; i++)
        {
            dat[i].x = locations->x[i];
            dat[i].y = locations->y[i];
            dat[i].w = w[i];
        }
        double mindist = 2, x_mean = 0, y_mean = 0;
        for (int i = 0; i < n; i++)
        {
            x_mean = x_mean + dat[i].x;
            y_mean = y_mean + dat[i].y;
        }
        x_mean = x_mean / n;
        y_mean = y_mean / n;

        for (int i = 0; i < n; i++)
        {
            flag[i] = 0;
            double dist = pow(dat[i].x - x_mean, 2) + pow(dat[i].y - y_mean, 2);
            if (dist < mindist)
            {
                mindist = dist;
                res[0] = i;
            }
        }
        flag[res[0]] = 1;

        for (int j = 1; j < n - 1; j++)
        {
            double max_list[n];
            for (int i = 0; i < n; i++)
            {
                if (flag[i] != 1)
                {
                    // double min_list[j];
                    double min_temp = 2;
                    for (int k = 0; k < j; k++)
                    {
                        double temp = pow(dat[i].x - dat[res[k]].x, 2) + pow(dat[i].y - dat[res[k]].y, 2);
                        if (temp < min_temp)
                            min_temp = temp;
                    }
                    max_list[i] = min_temp;
                }
                else
                    max_list[i] = 0;
            }
            double max_temp = 0;
            int ind_temp = n;
            for (int i = 0; i < n; i++)
            {
                if (max_temp < max_list[i])
                {
                    max_temp = max_list[i];
                    ind_temp = i;
                }
            }
            res[j] = ind_temp;
            flag[res[j]] = 1;
        }

        for (int i = 0; i < n; i++)
        {
            if (flag[i] != 1)
                res[n - 1] = i;
            locations->x[i] = dat[res[i]].x;
            locations->y[i] = dat[res[i]].y;
            w[i] = dat[res[i]].w;
        }
    }

    // 3D reordering
    struct comb_3d
    { // location and measure
        uint64_t Z;
        double w;
    };
    static uint64_t Part1By3(uint64_t x)
    // Spread lower bits of input
    {
        x &= 0x000000000000ffff;
        // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x << 24)) & 0x000000ff000000ff;
        // x = ---- ---- ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- ---- ---- 7654 3210
        x = (x ^ (x << 12)) & 0x000f000f000f000f;
        // x = ---- ---- ---- fedc ---- ---- ---- ba98 ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x ^ (x << 6)) & 0x0303030303030303;
        // x = ---- --fe ---- --dc ---- --ba ---- --98 ---- --76 ---- --54 ---- --32 ---- --10
        x = (x ^ (x << 3)) & 0x1111111111111111;
        // x = ---f ---e ---d ---c ---b ---a ---9 ---8 ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        return x;
    }
    static uint64_t EncodeMorton3(uint64_t x, uint64_t y, uint64_t z)
    // Encode 3 inputs into one
    {
        return (Part1By3(z) << 2) + (Part1By3(y) << 1) + Part1By3(x);
    }
    static uint64_t Compact1By3(uint64_t x)
    // Collect every 4-th bit into lower part of input
    {
        x &= 0x1111111111111111;
        // x = ---f ---e ---d ---c ---b ---a ---9 ---8 ---7 ---6 ---5 ---4 ---3 ---2 ---1 ---0
        x = (x ^ (x >> 3)) & 0x0303030303030303;
        // x = ---- --fe ---- --dc ---- --ba ---- --98 ---- --76 ---- --54 ---- --32 ---- --10
        x = (x ^ (x >> 6)) & 0x000f000f000f000f;
        // x = ---- ---- ---- fedc ---- ---- ---- ba98 ---- ---- ---- 7654 ---- ---- ---- 3210
        x = (x ^ (x >> 12)) & 0x000000ff000000ff;
        // x = ---- ---- ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 24)) & 0x000000000000ffff;
        // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
        return x;
    }
    static uint64_t DecodeMorton3X(uint64_t code)
    // Decode first input
    {
        return Compact1By3(code >> 0);
    }
    static uint64_t DecodeMorton3Y(uint64_t code)
    // Decode second input
    {
        return Compact1By3(code >> 1);
    }
    static uint64_t DecodeMorton3Z(uint64_t code)
    // Decode third input
    {
        return Compact1By3(code >> 2);
    }
    static uint64_t EncodeHilbert3(uint64_t x, uint64_t y, uint64_t z)
    {
        uint16_t M = 1 << 15, P, Q, t;
        for (Q = M; Q > 1; Q >>= 1)
        {
            P = Q - 1;
            if (x & Q)
                x ^= P;
            else
            {
                t = (x ^ x) & P;
                x ^= t;
                x ^= t;
            }
            if (y & Q)
                x ^= P;
            else
            {
                t = (x ^ y) & P;
                x ^= t;
                y ^= t;
            }
            if (z & Q)
                x ^= P;
            else
            {
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
        // printf("%" PRIu64 ", %" PRIu64 ", %" PRIu64 " \n", x, y, z);
        for (int i = 0; i < 16; i++)
        {
            res = x >> (15 - i) & 1;
            result |= res << (47 - i * 3);
            res = y >> (15 - i) & 1;
            result |= res << (47 - (i * 3 + 1));
            res = z >> (15 - i) & 1;
            result |= res << (47 - (i * 3 + 2));
        }
        return (result);
    }
    static uint64_t *DecodeHilbert3(uint64_t result)
    {
        uint64_t N = 2 << 15, P, Q, t;
        uint64_t x = 0, y = 0, z = 0;
        static uint64_t X[3] = {0, 0, 0};
        uint64_t res;
        result = result << 16;
        for (int i = 0; i < 16; i++)
        {
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
        for (Q = 2; Q != N; Q <<= 1)
        {
            P = Q - 1;
            if (z & Q)
                x ^= P;
            else
            {
                t = (x ^ z) & P;
                x ^= t;
                z ^= t;
            }
            if (y & Q)
                x ^= P;
            else
            {
                t = (x ^ y) & P;
                x ^= t;
                y ^= t;
            }
            if (x & Q)
                x ^= P;
            else
            {
                t = 0 & P;
                x ^= t;
                x ^= t;
            }
        }
        X[0] = x;
        X[1] = y;
        X[2] = z;
        return X;
    }
    static int cmpfunc_loc_3d(const void *a, const void *b)
    {
        struct comb_3d _a = *(const struct comb_3d *)a;
        struct comb_3d _b = *(const struct comb_3d *)b;
        if (_a.Z < _b.Z)
            return -1;
        if (_a.Z == _b.Z)
            return 0;
        return 1;
    }
    struct tree_3d
    {
        int dim;
        double x, y, z;
        double w;
        struct tree_3d *left, *right;
    };

    struct arr_3d
    {
        double x, y, z;
        double w;
    };

    static int cmpfunc_x_3d(const void *a, const void *b)
    {
        struct arr_3d _a = *(const struct arr_3d *)a;
        struct arr_3d _b = *(const struct arr_3d *)b;
        if (_a.x < _b.x)
            return -1;
        if (_a.x == _b.x)
            return 0;
        return 1;
    }

    static int cmpfunc_y_3d(const void *a, const void *b)
    {
        struct arr_3d _a = *(const struct arr_3d *)a;
        struct arr_3d _b = *(const struct arr_3d *)b;
        if (_a.y < _b.y)
            return -1;
        if (_a.y == _b.y)
            return 0;
        return 1;
    }

    static int cmpfunc_z_3d(const void *a, const void *b)
    {
        struct arr_3d _a = *(const struct arr_3d *)a;
        struct arr_3d _b = *(const struct arr_3d *)b;
        if (_a.z < _b.z)
            return -1;
        if (_a.z == _b.z)
            return 0;
        return 1;
    }

    static struct tree_3d *initialize_3d(int len, struct arr_3d *dat, int depth)
    {
        if (len < 1)
            return NULL;
        if (len == 1)
        {
            struct tree_3d *root = (struct tree_3d *)malloc(sizeof(struct tree_3d));
            root->dim = -1;
            root->x = dat[0].x;
            root->y = dat[0].y;
            root->z = dat[0].z;
            root->w = dat[0].w;
            root->left = NULL;
            root->right = NULL;
            return root;
        }
        struct tree_3d *root = (struct tree_3d *)malloc(sizeof(struct tree_3d));
        // int dim = (x[end] - x[start] > y[end] - y[start]) ? 1 : 2;
        int dim = depth % 3;
        root->dim = dim;
        int mid = len / 2;
        // printf("length: %d, mid: %f\n", len, dat[mid].x);
        if (dim == 0)
            qsort(dat, len, sizeof(struct arr_3d), cmpfunc_x_3d);
        else if (dim == 1)
            qsort(dat, len, sizeof(struct arr_3d), cmpfunc_y_3d);
        else
            qsort(dat, len, sizeof(struct arr_3d), cmpfunc_z_3d);
        // printf("length: %d, mid: %f\n", len, dat[mid].x);
        if (dim == 0)
            root->x = dat[mid].x;
        else if (dim == 1)
            root->y = dat[mid].y;
        else if (dim == 2)
            root->z = dat[mid].z;
        struct arr_3d *ldat = (struct arr_3d *)malloc(mid * sizeof(struct arr_3d));
        struct arr_3d *rdat = (struct arr_3d *)malloc((len - mid) * sizeof(struct arr_3d));
        memcpy(ldat, dat, mid * sizeof(struct arr_3d));
        memcpy(rdat, dat + mid, (len - mid) * sizeof(struct arr_3d));
        free(dat);
        root->left = initialize_3d(mid, ldat, depth + 1);
        root->right = initialize_3d(len - mid, rdat, depth + 1);
        return root;
    }

    static int traverse_3d(struct tree_3d *root, int i, double *x, double *y, double *z, double *w)
    {
        if (root->left != NULL)
            i = traverse_3d(root->left, i, x, y, z, w);
        if (root->dim == -1)
        {
            // printf("%d\n", i);
            x[i] = root->x;
            y[i] = root->y;
            z[i] = root->z;
            w[i++] = root->w;
        }
        if (root->right != NULL)
            i = traverse_3d(root->right, i, x, y, z, w);
        return i;
    }

    // random ordering 3D
    static void random_reordering_3d(int size, location *loc)
    {
        int your_seed_value = 42; // Set your desired seed value

        srand(your_seed_value);

        for (int i = size - 1; i > 0; i--)
        {
            int j = rand() % (i + 1);
            // Swap x values
            double tempX = loc->x[i];
            loc->x[i] = loc->x[j];
            loc->x[j] = tempX;

            // Swap y values
            double tempY = loc->y[i];
            loc->y[i] = loc->y[j];
            loc->y[j] = tempY;

            // Swap z values
            double tempZ = loc->z[i];
            loc->z[i] = loc->z[j];
            loc->z[j] = tempZ;

            // // Swap obs values
            // double tempObs = h_C[i];
            // h_C[i] = h_C[j];
            // h_C[j] = tempObs;
        }
    }

    // morton 3D
    static void zsort_locations_morton_3d(int n, location *locations)
    //! Sort in Morton order 3d (input points must be in [0;1]x[0;1]x[0;1] sphere])
    {
        int i;
        uint16_t x, y, z;
        struct comb_3d *dat = (struct comb_3d *)malloc(n * sizeof(struct comb_3d));
        //  Encode data into vector z
        for (i = 0; i < n; i++)
        {
            x = (uint16_t)(locations->x[i] * (double)UINT16_MAX + .5);
            y = (uint16_t)(locations->y[i] * (double)UINT16_MAX + .5);
            z = (uint16_t)(locations->z[i] * (double)UINT16_MAX + .5);
            dat[i].Z = EncodeMorton3(x, y, z);
        }
        // Sort vector z
        qsort(dat, n, sizeof(struct comb_3d), cmpfunc_loc_3d);
        // Decode data from vector z
        for (i = 0; i < n; i++)
        {
            x = DecodeMorton3X(dat[i].Z);
            y = DecodeMorton3Y(dat[i].Z);
            z = DecodeMorton3Z(dat[i].Z);
            locations->x[i] = (double)x / (double)UINT16_MAX;
            locations->y[i] = (double)y / (double)UINT16_MAX;
            locations->z[i] = (double)z / (double)UINT16_MAX;
        }
    }

    // hilbert 3D
    static void zsort_locations_hilbert_3d(int n, location *locations)
    {
        int i;
        uint16_t x, y, z;
        struct comb_3d *dat = (struct comb_3d *)malloc(n * sizeof(struct comb_3d));
        for (i = 0; i < n; i++)
        {
            x = (uint16_t)(locations->x[i] * (double)UINT16_MAX + .5);
            y = (uint16_t)(locations->y[i] * (double)UINT16_MAX + .5);
            z = (uint16_t)(locations->z[i] * (double)UINT16_MAX + .5);
            dat[i].Z = EncodeHilbert3(x, y, z);
        }
        //  Sort vector z
        qsort(dat, n, sizeof(struct comb_3d), cmpfunc_loc_3d);
        // Decode data from vector z
        for (i = 0; i < n; i++)
        {
            uint64_t *X = DecodeHilbert3(dat[i].Z);
            x = *X;
            y = *(X + 1);
            z = *(X + 2);
            locations->x[i] = (double)x / (double)UINT16_MAX;
            locations->y[i] = (double)y / (double)UINT16_MAX;
            locations->z[i] = (double)z / (double)UINT16_MAX;
        }
    }

    // kdtree 3D
    static void zsort_locations_kdtree_3d(int n, location *locations)
    {
        int i;
        struct arr_3d *dat = (struct arr_3d *)malloc(n * sizeof(struct arr_3d));
        double x[n], y[n], z[n];
        double *w = (double *)malloc(n * sizeof(double));

        // Encode data into vector z
        for (i = 0; i < n; i++)
        {
            dat[i].x = locations->x[i];
            dat[i].y = locations->y[i];
            dat[i].z = locations->z[i];
        }
        struct tree_3d *root = initialize_3d(n, dat, 0);
        int index = traverse_3d(root, 0, x, y, z, w);

        for (i = 0; i < n; i++)
        {
            locations->x[i] = x[i];
            locations->y[i] = y[i];
            locations->z[i] = z[i];
        }
    }

    // mmd 3D
    static void zsort_locations_mmd_3d(int n, location *locations)
    {
        int res[n], flag[n];
        // double *w = (double *)malloc(n * sizeof(double));
        struct arr_3d *dat = (struct arr_3d *)malloc(n * sizeof(struct arr_3d));
        int n_measurement = 1;

        for (int i = 0; i < n; i++)
        {
            dat[i].x = locations->x[i];
            dat[i].y = locations->y[i];
            dat[i].z = locations->z[i];
            // dat[i].w = w[i];
        }
        double mindist = 3, x_mean = 0, y_mean = 0, z_mean = 0;
        for (int i = 0; i < n; i++)
        {
            x_mean = x_mean + dat[i].x;
            y_mean = y_mean + dat[i].y;
            z_mean = z_mean + dat[i].z;
        }
        x_mean = x_mean / n;
        y_mean = y_mean / n;
        z_mean = z_mean / n;

        for (int i = 0; i < n; i++)
        {
            flag[i] = 0;
            double dist = pow(dat[i].x - x_mean, 2) + pow(dat[i].y - y_mean, 2) + pow(dat[i].z - z_mean, 2);
            if (dist < mindist)
            {
                mindist = dist;
                res[0] = i;
            }
        }
        flag[res[0]] = 1;

        for (int j = 1; j < n - 1; j++)
        {
            double max_list[n];
            for (int i = 0; i < n; i++)
            {
                if (flag[i] != 1)
                {
                    // double min_list[j];
                    double min_temp = 3;
                    for (int k = 0; k < j; k++)
                    {
                        double temp = pow(dat[i].x - dat[res[k]].x, 2) + pow(dat[i].y - dat[res[k]].y, 2) + pow(dat[i].z - dat[res[k]].z, 2);
                        if (temp < min_temp)
                            min_temp = temp;
                    }
                    max_list[i] = min_temp;
                }
                else
                    max_list[i] = 0;
            }
            double max_temp = 0;
            int ind_temp = n;
            for (int i = 0; i < n; i++)
            {
                if (max_temp < max_list[i])
                {
                    max_temp = max_list[i];
                    ind_temp = i;
                }
            }
            res[j] = ind_temp;
            flag[res[j]] = 1;
        }

        for (int i = 0; i < n; i++)
        {
            if (flag[i] != 1)
                res[n - 1] = i;
            locations->x[i] = dat[res[i]].x;
            locations->y[i] = dat[res[i]].y;
            locations->z[i] = dat[res[i]].z;
            // w[i] = dat[res[i]].w;
        }
    }

    // Generate the covariance matrix.
    void core_scmg(float *A, int m, int n,
                   int m0, int n0,
                   location *l1, location *l2,
                   double *localtheta, int distance_metric);

    void core_dcmg(double *A, int m, int n,
                   //    int m0, int n0,
                   location *l1, location *l2,
                   const double *localtheta, int distance_metric,
                   int z_flag, double dist_scale);

    void core_dcmg_nugget(double *A, int m, int n,
                           //    int m0, int n0,
                           location *l1, location *l2,
                           const double *localtheta, int distance_metric,
                           int z_flag, double dist_scale);

    void core_sdcmg(double *A, int m, int n,
                    int m0, int n0,
                    location *l1, location *l2,
                    double *localtheta, int distance_metric);

    void core_scmg_pow_exp(float *A, int m, int n,
                           int m0, int n0,
                           location *l1, location *l2,
                           double *localtheta, int distance_metric);

    void core_dcmg_pow_exp(double *A, int m, int n,
                           //    int m0, int n0,
                           location *l1, location *l2,
                           const double *localtheta, int distance_metric);

    void core_sdcmg_pow_exp(double *A, int m, int n,
                            int m0, int n0,
                            location *l1, location *l2,
                            double *localtheta, int distance_metric);

    // void core_dcmg_bivariate_parsimonious(double* A, int m, int n,
    //                                       int m0, int n0, location *l1,
    //                                       location *l2, double* localtheta, int distance_metric);
    void core_dcmg_bivariate_parsimonious(double *A, int m, int n,
                                          //   int m0, int n0,
                                          location *l1,
                                          location *l2, const double *localtheta, int distance_metric);

    void core_dcmg_bivariate_parsimonious2(double *A, int m, int n,
                                           int m0, int n0, location *l1,
                                           location *l2, double *localtheta, int distance_metric, int size);

    void core_dcmg_bivariate_flexible(double *A, int m, int n,
                                      int m0, int n0, location *l1,
                                      location *l2, double *localtheta, int distance_metric);

    float core_smdet(float *A, int m, int n,
                     int m0, int n0);

    double core_dmdet(double *A, int m, int n,
                      int m0, int n0);

    void core_szcpy(float *Z, int m,
                    int m0, float *r);

    void core_dzcpy(double *Z, int m,
                    int m0, double *r);

    float core_sdotp(float *Z, float *dotproduct,
                     int n);

    double core_ddotp(double *Z, double *dotproduct,
                      int n);

    void core_dlag2s(int m, int n,
                     const double *A, int lda,
                     float *B, int ldb);

    void core_slag2d(int m, int n,
                     const float *A, int lda,
                     double *B, int ldb);

    void core_sprint(float *A,
                     int m, int n,
                     int m0, int n0);

    void core_dprint(double *A,
                     int m, int n,
                     int m0, int n0);

    void core_dcmg_nono_stat(double *A, int m, int n,
                             int m0, int n0, location *l1,
                             location *l2, location *lm, double *localtheta,
                             int distance_metric);

    void core_dcmg_matern_dsigma_square(double *A, int m, int n,
                                        int m0, int n0, location *l1,
                                        location *l2, double *localtheta, int distance_metric);

    void core_dcmg_matern_dnu(double *A, int m, int n,
                              int m0, int n0, location *l1,
                              location *l2, double *localtheta, int distance_metric);

    void core_dcmg_matern_dbeta(double *A, int m, int n,
                                int m0, int n0, location *l1,
                                location *l2, double *localtheta, int distance_metric);

    void core_dcmg_matern_ddsigma_square(double *A, int m, int n);

    void core_dcmg_matern_ddsigma_square_beta(double *A, int m, int n,
                                              int m0, int n0, location *l1,
                                              location *l2, double *localtheta, int distance_metric);

    void core_dcmg_matern_ddsigma_square_nu(double *A, int m, int n,
                                            int m0, int n0, location *l1,
                                            location *l2, double *localtheta, int distance_metric);

    void core_dcmg_matern_ddbeta_beta(double *A, int m, int n,
                                      int m0, int n0, location *l1,
                                      location *l2, double *localtheta, int distance_metric);

    void core_dcmg_matern_ddbeta_nu(double *A, int m, int n,
                                    int m0, int n0, location *l1,
                                    location *l2, double *localtheta, int distance_metric);

    double core_dtrace(double *A, int m, int n,
                       int m0, int n0, double *trace);

    double core_ng_loglike(double *Z, double *localtheta,
                           int m);

    void core_ng_transform(double *Z, double *nan_flag, double *localtheta,
                           int m);

    void core_g_to_ng(double *Z, double *localtheta,
                      int m);

    double core_dtrace(double *A, int m, int n,
                       int m0, int n0, double *trace);

    void core_dcmg_nuggets(double *A, int m, int n,
                           int m0, int n0, location *l1,
                           location *l2, double *localtheta, int distance_metric);

    void core_dcmg_spacetime_bivariate_parsimonious(double *A, int m, int n,
                                                    int m0, int n0, location *l1,
                                                    location *l2, double *localtheta, int distance_metric);

    void core_dcmg_non_stat(double *A, int m, int n, int m0,
                            int n0, location *l1, location *l2, double *localtheta, int distance_metric);

    void core_dcmg_spacetime_matern(double *A, int m, int n,
                                    location *l1,
                                    location *l2, const double *localtheta, int distance_metric);

    void core_dcmg_matern_ddnu_nu(double *A, int m, int n, int m0, int n0, location *l1, location *l2, double *localtheta,
                                  int distance_metric);

    void core_ng_dcmg(double *A, int m, int n,
                      int m0, int n0, location *l1,
                      location *l2, double *localtheta, int distance_metric);

    void core_ng_exp_dcmg(double *A, int m, int n,
                          int m0, int n0, location *l1,
                          location *l2, double *localtheta, int distance_metric);

    void core_dcmg_trivariate_parsimonious(double *A, int m, int n,
                                           int m0, int n0, location *l1,
                                           location *l2, double *localtheta, int distance_metric);
    // Generate the covariance matrix.
    void core_dcmg_matern12(double *A, int m, int n,
                            location *l1, location *l2,
                            const double *localtheta);
#ifdef __cplusplus
}
#endif
#endif
