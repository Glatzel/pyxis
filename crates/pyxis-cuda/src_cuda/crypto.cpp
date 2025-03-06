#pragma once
#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#include <cuda_runtime.h>
#else
#define CUDA_DEVICE // Empty for normal C++ compilation
#include <cmath>
#endif
#define M_PI 3.14159265358979323846264
#define EE 0.006693421622965943
#define krasovsky1940_A 6378245.0
CUDA_DEVICE void transform(
    const double x, const double y,
    double &lon, double &lat)
{
    double xy = x * y;
    double abs_x = sqrt(abs(x));
    double x_pi = x * M_PI;
    double y_pi = y * M_PI;
    double d = 20.0 * sin(6.0 * x_pi) + 20.0 * sin(2.0 * x_pi);

    lat = d;
    lon = d;

    lat += 20.0 * sin(y_pi) + 40.0 * sin(y_pi / 3.0) + 160.0 * sin(y_pi / 12.0) + 320.0 * sin(y_pi / 30.0);
    lon += 20.0 * sin(x_pi) + 40.0 * sin(x_pi / 3.0) + 150.0 * sin(x_pi / 12.0) + 300.0 * sin(x_pi / 30.0);

    lat *= 2.0 / 3.0;
    lon *= 2.0 / 3.0;

    lat += -100.0 + 2.0 * x + 3.0 * y + 0.2 * pow(y, 2) + 0.1 * xy + 0.2 * abs_x;
    lon += 300.0 + x + 2.0 * y + 0.1 * pow(x, 2) + 0.1 * xy + 0.1 * abs_x;
}
CUDA_DEVICE void delta(
    const double lon, const double lat,
    double &d_lon, double &d_lat)
{

    transform(lon - 105.0, lat - 35.0, d_lon, d_lat);

    double rad_lat = lat / 180.0 * M_PI;
    double magic = sin(rad_lat);
    double earth_r = krasovsky1940_A;

    magic = 1.0 - EE * magic * magic;
    double sqrt_magic = sqrt(magic);

    d_lat = (d_lat * 180.0) / ((earth_r * (1.0 - EE)) / (magic * sqrt_magic) * M_PI);
    d_lon = (d_lon * 180.0) / (earth_r / sqrt_magic * cos(rad_lat) * M_PI);
}
CUDA_DEVICE void bd09_to_gcj02(
    const double bd09_lon, const double bd09_lat,
    double &gcj02_lon, double &gcj02_lat)
{
    double x_pi = M_PI * 3000.0 / 180.0;
    double x = bd09_lon - 0.0065;
    double y = bd09_lat - 0.006;
    double z = sqrt(pow(x, 2) + pow(y, 2)) - 0.00002 * sin(y * x_pi);
    double theta = atan2(y, x) - 0.000003 * cos(x * x_pi);
    gcj02_lon = z * cos(theta);
    gcj02_lat = z * sin(theta);
}
CUDA_DEVICE void gcj02_to_bd09(
    const double gcj02_lon, const double gcj02_lat,
    double &bd09_lon, double &bd09_lat)
{
    double x_pi = M_PI * 3000.0 / 180.0;
    double z = sqrt(pow(gcj02_lon, 2) + pow(gcj02_lat, 2)) + 0.00002 * sin(gcj02_lat * x_pi);
    double theta = atan2(gcj02_lat, gcj02_lon) + 0.000003 * cos(gcj02_lon * x_pi);
    bd09_lon = z * cos(theta) + 0.0065;
    bd09_lat = z * sin(theta) + 0.006;
}
CUDA_DEVICE void gcj02_to_wgs84(
    const double gcj02_lon, const double gcj02_lat,
    double &wgs84_lon, double &wgs84_lat)
{
    double d_lon = 0.0;
    double d_lat = 0.0;
    delta(gcj02_lon, gcj02_lat, d_lon, d_lat);
    wgs84_lon = gcj02_lon - d_lon;
    wgs84_lat = gcj02_lat - d_lat;
}
CUDA_DEVICE void wgs84_to_gcj02(
    const double wgs84_lon, const double wgs84_lat,
    double &gcj02_lon, double &gcj02_lat)
{
    double d_lon = 0.0;
    double d_lat = 0.0;
    delta(wgs84_lon, wgs84_lat, d_lon, d_lat);
    gcj02_lon = wgs84_lon + d_lon;
    gcj02_lat = wgs84_lat + d_lat;
}
CUDA_DEVICE void wgs84_to_bd09(
    const double wgs84_lon, const double wgs84_lat,
    double &bd09_lon, double &bd09_lat)
{
    wgs84_to_gcj02(wgs84_lon, wgs84_lat,
                   bd09_lon, bd09_lat);
    gcj02_to_bd09(bd09_lon, bd09_lat,
                  bd09_lon, bd09_lat);
}
CUDA_DEVICE void bd09_to_wgs84(
    const double bd09_lon, const double bd09_lat,
    double &wgs84_lon, double &wgs84_lat)
{
    bd09_to_gcj02(bd09_lon, bd09_lat,
                  wgs84_lon, wgs84_lat);
    gcj02_to_wgs84(wgs84_lon, wgs84_lat,
                   wgs84_lon, wgs84_lat);
}
CUDA_DEVICE double to_radians(const double degrees)
{
    return degrees * M_PI / 180.0;
}
CUDA_DEVICE double haversine_distance(const double lon_a, const double lat_a,
                                      const double lon_b, const double lat_b)
{
    // Convert latitudes and longitudes to radians
    double lat1_rad = to_radians(lat_a);
    double lon1_rad = to_radians(lon_a);
    double lat2_rad = to_radians(lat_b);
    double lon2_rad = to_radians(lon_b);

    // Calculate differences
    double delta_lat = lat2_rad - lat1_rad;
    double delta_lon = lon2_rad - lon1_rad;

    // Haversine formula
    double a = pow(sin(delta_lat / 2.0), 2) +
               cos(lat1_rad) * cos(lat2_rad) *
                   pow(sin(delta_lon / 2.0), 2);

    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return 6378137.0 * c;
}
CUDA_DEVICE void crypto_exact(
    const double src_lon,
    const double src_lat,
    void (*crypto_fn)(const double, const double, double &, double &),
    void (*inv_crypto_fn)(const double, const double, double &, double &),
    const double threshold,
    const bool distance_mode,
    const int max_iter,
    double &out_lon,
    double &out_lat)

{
    double dst_lon = src_lon;
    double dst_lat = src_lat;
    crypto_fn(src_lon, src_lat, dst_lon, dst_lat);
    for (int i = 0; i < max_iter; i++)
    {
        double tmp_src_lon = 0.0;
        double tmp_src_lat = 0.0;
        inv_crypto_fn(dst_lon, dst_lat, tmp_src_lon, tmp_src_lat);
        double d_lon = src_lon - tmp_src_lon;
        double d_lat = src_lat - tmp_src_lat;
        double tmp_lon = dst_lon + d_lon;
        double tmp_lat = dst_lat + d_lat;

        if (distance_mode)
        {
            if (haversine_distance(dst_lon, dst_lat, tmp_lon, tmp_lat) < threshold)
            {
                break;
            }
        }
        else
        {
            if (abs(src_lon - tmp_lon) < threshold && abs(src_lat - tmp_lat) < threshold)
            {
                break;
            }
        }
        dst_lon = tmp_lon;
        dst_lat = tmp_lat;
    }
    out_lon = dst_lon;
    out_lat = dst_lat;
}
