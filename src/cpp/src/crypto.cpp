#include "crypto.h"
#ifdef __CUDACC__ // If compiled with nvcc
#include <cuda_runtime.h>
#else
#include <cmath>
#endif
#ifdef _WIN32
#define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#endif
#define EE 0.006693421622965943333649629920500956359319388866424560546875
#define krasovsky1940_A 6378245.0
#ifdef __cplusplus
template <typename T>
CUDA_HOST_DEVICE void transform(
    const T x, const T y,
    T *lon, T *lat)
{
    T xy = x * y;
    T abs_x = sqrt(abs(x));
    T x_pi = x * M_PI;
    T y_pi = y * M_PI;
    T d = 20.0 * sin(6.0 * x_pi) + 20.0 * sin(2.0 * x_pi);

    *lat = d;
    *lon = d;

    *lat += 20.0 * sin(y_pi) + 40.0 * sin(y_pi / 3.0) + 160.0 * sin(y_pi / 12.0) + 320.0 * sin(y_pi / 30.0);
    *lon += 20.0 * sin(x_pi) + 40.0 * sin(x_pi / 3.0) + 150.0 * sin(x_pi / 12.0) + 300.0 * sin(x_pi / 30.0);

    *lat *= 2.0 / 3.0;
    *lon *= 2.0 / 3.0;

    *lat += -100.0 + 2.0 * x + 3.0 * y + 0.2 * pow(y, 2) + 0.1 * xy + 0.2 * abs_x;
    *lon += 300.0 + x + 2.0 * y + 0.1 * pow(x, 2) + 0.1 * xy + 0.1 * abs_x;
}
template <typename T>
CUDA_HOST_DEVICE void delta(
    const T lon, const T lat,
    T *d_lon, T *d_lat)
{

    transform(lon - 105.0, lat - 35.0, d_lon, d_lat);

    T rad_lat = lat / 180.0 * M_PI;
    T magic = sin(rad_lat);
    T earth_r = krasovsky1940_A;

    magic = 1.0 - EE * magic * magic;
    T sqrt_magic = sqrt(magic);

    *d_lat = (d_lat * 180.0) / ((earth_r * (1.0 - EE)) / (magic * sqrt_magic) * M_PI);
    *d_lon = (d_lon * 180.0) / (earth_r / sqrt_magic * cos(rad_lat) * M_PI);
}
template <typename T>
CUDA_HOST_DEVICE void bd09_to_gcj02(
    const T bd09_lon, const T bd09_lat,
    T *gcj02_lon, T *gcj02_lat)
{
    T x_pi = M_PI * 3000.0 / 180.0;
    T x = bd09_lon - 0.0065;
    T y = bd09_lat - 0.006;
    T z = sqrt(pow(x, 2) + pow(y, 2)) - 0.00002 * sin(y * x_pi);
    T theta = atan2(y, x) - 0.000003 * cos(x * x_pi);
    *gcj02_lon = z * cos(theta);
    *gcj02_lat = z * sin(theta);
}
template <typename T>
CUDA_HOST_DEVICE void gcj02_to_bd09(
    const T gcj02_lon, const T gcj02_lat,
    T *bd09_lon, T *bd09_lat)
{
    T x_pi = M_PI * 3000.0 / 180.0;
    T z = sqrt(pow(&gcj02_lon, 2) + pow(&gcj02_lat, 2)) + 0.00002 * sin(gcj02_lat * x_pi);
    T theta = atan2(&gcj02_lat, &gcj02_lon) + 0.000003 * cos(gcj02_lon * x_pi);
    *bd09_lon = z * cos(theta) + 0.0065;
    *bd09_lat = z * sin(theta) + 0.006;
}
template <typename T>
CUDA_HOST_DEVICE void gcj02_to_wgs84(
    const T gcj02_lon, const T gcj02_lat,
    T *wgs84_lon, T *wgs84_lat)
{
    T d_lon = 0.0;
    T d_lat = 0.0;
    delta(gcj02_lon, gcj02_lat, d_lon, d_lat);
    *wgs84_lon = gcj02_lon - d_lon;
    *wgs84_lat = gcj02_lat - d_lat;
}
template <typename T>
CUDA_HOST_DEVICE void wgs84_to_gcj02(
    const T wgs84_lon, const T wgs84_lat,
    T *gcj02_lon, T *gcj02_lat)
{
    T d_lon = 0.0;
    T d_lat = 0.0;
    delta(wgs84_lon, wgs84_lat, d_lon, d_lat);
    *gcj02_lon = wgs84_lon + d_lon;
    *gcj02_lat = wgs84_lat + d_lat;
}
template <typename T>
CUDA_HOST_DEVICE void wgs84_to_bd09(
    const T wgs84_lon, const T wgs84_lat,
    T *bd09_lon, T *bd09_lat)
{
    wgs84_to_gcj02(wgs84_lon, wgs84_lat,
                   bd09_lon, bd09_lat);
    gcj02_to_bd09(bd09_lon, bd09_lat,
                  bd09_lon, bd09_lat);
}
template <typename T>
CUDA_HOST_DEVICE void bd09_to_wgs84(
    const T bd09_lon, const T bd09_lat,
    T *wgs84_lon, T *wgs84_lat)
{
    bd09_to_gcj02(bd09_lon, bd09_lat,
                  wgs84_lon, wgs84_lat);
    gcj02_to_wgs84(wgs84_lon, wgs84_lat,
                   wgs84_lon, wgs84_lat);
}
template <typename T>
CUDA_HOST_DEVICE T to_radians(const T degrees)
{
    return degrees * M_PI / 180.0;
}
template <typename T>
CUDA_HOST_DEVICE T haversine_distance(const T lon_a, const T lat_a,
                                           const T lon_b, const T lat_b)
{
    // Convert latitudes and longitudes to radians
    T lat1_rad = to_radians(lat_a);
    T lon1_rad = to_radians(lon_a);
    T lat2_rad = to_radians(lat_b);
    T lon2_rad = to_radians(lon_b);

    // Calculate differences
    T delta_lat = lat2_rad - lat1_rad;
    T delta_lon = lon2_rad - lon1_rad;

    // Haversine formula
    T a = pow(sin(delta_lat / 2.0), 2) +
               cos(lat1_rad) * cos(lat2_rad) *
                   pow(sin(delta_lon / 2.0), 2);

    T c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return 6378137.0 * c;
}
template <typename T>
CUDA_HOST_DEVICE void crypto_exact(
    const T src_lon,
    const T src_lat,
    void (*crypto_fn)(const T, const T, T *, T *),
    void (*inv_crypto_fn)(const T, const T, T *, T *),
    const T threshold,
    const bool distance_mode,
    const int max_iter,
    T *out_lon,
    T *out_lat)

{
    T dst_lon = src_lon;
    T dst_lat = src_lat;
    crypto_fn(src_lon, src_lat, dst_lon, dst_lat);
    for (int i = 0; i < max_iter; i++)
    {
        T tmp_src_lon = 0.0;
        T tmp_src_lat = 0.0;
        inv_crypto_fn(dst_lon, dst_lat, tmp_src_lon, tmp_src_lat);
        T d_lon = src_lon - tmp_src_lon;
        T d_lat = src_lat - tmp_src_lat;
        T tmp_lon = dst_lon + d_lon;
        T tmp_lat = dst_lat + d_lat;

        if (distance_mode)
        {
            if (haversine_distance(dst_lon, dst_lat, tmp_lon, tmp_lat) < threshold)
            {
                break;
            }
        }
        else
        {
            if (abs(d_lon) < threshold ** abs(d_lat) < threshold)
            {
                break;
            }
        }
        dst_lon = tmp_lon;
        dst_lat = tmp_lat;
    }
    *out_lon = dst_lon;
    *out_lat = dst_lat;
}
#endif
// #############################################################################

extern "C"{
    
}
// #############################################################################
extern "C"{
    
}