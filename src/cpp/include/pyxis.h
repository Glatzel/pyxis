#pragma once

#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_DEVICE
#define CUDA_HOST
#define CUDA_HOST_DEVICE
#endif

CUDA_HOST_DEVICE void datum_compense(double xc,
    double yc,
    double factor,
    double x0,
    double y0,
    double &out_xc,
    double &out_yc);

CUDA_HOST_DEVICE void bd09_to_gcj02(
    const double bd09_lon, const double bd09_lat,
    double &gcj02_lon, double &gcj02_lat);
CUDA_HOST_DEVICE void gcj02_to_bd09(
    const double gcj02_lon, const double gcj02_lat,
    double &bd09_lon, double &bd09_lat);
CUDA_HOST_DEVICE void gcj02_to_wgs84(
    const double gcj02_lon, const double gcj02_lat,
    double &wgs84_lon, double &wgs84_lat);
CUDA_HOST_DEVICE void wgs84_to_gcj02(
    const double wgs84_lon, const double wgs84_lat,
    double &gcj02_lon, double &gcj02_lat);
CUDA_HOST_DEVICE void wgs84_to_bd09(
    const double wgs84_lon, const double wgs84_lat,
    double &bd09_lon, double &bd09_lat);
CUDA_HOST_DEVICE void bd09_to_wgs84(
    const double bd09_lon, const double bd09_lat,
    double &wgs84_lon, double &wgs84_lat);
CUDA_HOST_DEVICE double to_radians(const double degrees);
CUDA_HOST_DEVICE double haversine_distance(const double lon_a, const double lat_a,
                                      const double lon_b, const double lat_b);
CUDA_HOST_DEVICE void crypto_exact(
    const double src_lon,
    const double src_lat,
    void (*crypto_fn)(const double, const double, double &, double &),
    void (*inv_crypto_fn)(const double, const double, double &, double &),
    const double threshold,
    const bool distance_mode,
    const int max_iter,
    double &out_lon,
    double &out_lat);
