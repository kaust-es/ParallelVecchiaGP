#ifndef NEAREST_NEIGHBOR_H
#define NEAREST_NEIGHBOR_H

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#define pow_e (2.71828182845904)
#define PI (3.141592653589793)
#define earthRadiusKm (6371.0)

// used for nearest neighbor selection
double calEucDistance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}


/**
 * This function converts decimal degrees to radians
 * @param deg decimal degree
 */
static double deg2rad(double deg) {
	return (deg * PI / 180);
}

/**
 * This function converts radians to decimal degrees
 * @param rad radians
 */
static double rad2deg(double rad) {
	return (rad * 180 / PI);
}

/**
 * Returns the distance between two points on the Earth.
 * Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
 * @param lat1d Latitude of the first point in degrees
 * @param lon1d Longitude of the first point in degrees
 * @param lat2d Latitude of the second point in degrees
 * @param lon2d Longitude of the second point in degrees
 * @return The distance between the two points in kilometers
 */
static double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
	double lat1r, lon1r, lat2r, lon2r, u, v;
	lat1r = deg2rad(lat1d);
	lon1r = deg2rad(lon1d);
	lat2r = deg2rad(lat2d);
	lon2r = deg2rad(lon2d);
	u = sin((lat2r - lat1r) / 2);
	v = sin((lon2r - lon1r) / 2);
	return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

void findNearestPoints(double *h_C_conditioned, double *h_C, location* locations_con, location* locations, int l0, int l1, int l2, int k, int i_block, int distance_metric) {
    // For example, p(y_{6, 7, 8, 9}|y_{2, 3, 4, 5}) -> p(y1|y2)
    // l0: starting point, 0 or 1 in the example,affecting how large conditioning set you will choose
    // l1: the end of conditioning set, e.g., 5
    // l2: the end of conditioned set, e.g., 9
    // l2 - l1, conditioned set, y1
    // l1 - l0, conditioning set, y2, have not been sorted yet
    // int k: the conditioning size
    // int i_block: i th NN, 0 means the first vecchia approximation.
    //              because independent block has been already copied at first.
    // std::vector<std::pair<double, double>> nearestPoints;


    std::vector<double> distances;
    for (int i = l0; i < l1; i++) {
        double distance;
        if (distance_metric == 1){
            distance = distanceEarth(locations->x[l1], locations->y[l1],
                                        locations->x[i], locations->y[i]);
        }else{
            distance = calEucDistance(locations->x[l1], locations->y[l1],
                                        locations->x[i], locations->y[i]);
        }
        
        distances.push_back(distance);
    }

    std::vector<int> indices(l1 - l0);
    for (int i = l0; i < l1; i++) {
        indices[i - l0] = i;
    }
    // Sort only the first k distances
    // std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int i1, int i2) {
    //     return distances[i1] < distances[i2];
    // });
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
        return distances[i1] < distances[i2];
    });


    if (k > (l1 - l0)) {
        std::cout << "Not enough points available." << std::endl;
        k = l1 - l0;
    }
    // printf("=======(%lf, %lf) =======\n", locations->x[l1], locations->y[l1]);
    // printf("-----------------%d------------------\n", l1);
    for (int i = 0; i < k; i++) {
        // printf("(%lf, %lf)\n", locations->x[indices[i]], locations->y[indices[i]]);
        // printf("%d \n", indices[i]);
        locations_con->x[i_block * k + i] = locations->x[indices[i]];
        locations_con->y[i_block * k + i] = locations->y[indices[i]];
        // skip the first one, which is copied already
        h_C_conditioned[(i_block + 1) * k + i] = h_C[indices[i]];
        // printf("(%lf, %lf) \n", locations_con->x[l1 - 1 - i], locations_con->y[l1 - 1 - i]);
    }
}

#endif