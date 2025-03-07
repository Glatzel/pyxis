#pragma once
#include "cuda_macro.h"

CUDA_DEVICE void bd09_to_gcj02(
    const double bd09_lon, const double bd09_lat,
    double &gcj02_lon, double &gcj02_lat);
CUDA_DEVICE void gcj02_to_bd09(
    const double gcj02_lon, const double gcj02_lat,
    double &bd09_lon, double &bd09_lat);
CUDA_DEVICE void gcj02_to_wgs84(
    const double gcj02_lon, const double gcj02_lat,
    double &wgs84_lon, double &wgs84_lat);
CUDA_DEVICE void wgs84_to_gcj02(
    const double wgs84_lon, const double wgs84_lat,
    double &gcj02_lon, double &gcj02_lat);
CUDA_DEVICE void wgs84_to_bd09(
    const double wgs84_lon, const double wgs84_lat,
    double &bd09_lon, double &bd09_lat);
CUDA_DEVICE void bd09_to_wgs84(
    const double bd09_lon, const double bd09_lat,
    double &wgs84_lon, double &wgs84_lat);
CUDA_DEVICE double to_radians(const double degrees);
CUDA_DEVICE double haversine_distance(const double lon_a, const double lat_a,
                                      const double lon_b, const double lat_b);
CUDA_DEVICE void crypto_exact(
    const double src_lon,
    const double src_lat,
    void (*crypto_fn)(const double, const double, double &, double &),
    void (*inv_crypto_fn)(const double, const double, double &, double &),
    const double threshold,
    const bool distance_mode,
    const int max_iter,
    double &out_lon,
    double &out_lat);