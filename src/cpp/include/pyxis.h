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


// datum_compense.cpp
template <typename T>
CUDA_HOST_DEVICE void datum_compense(T xc,
    T yc,
    T factor,
    T x0,
    T y0,
    T &out_xc,
    T &out_yc);

// crypto.cpp
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
