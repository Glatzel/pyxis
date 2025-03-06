#pragma once
#ifdef __CUDACC__ // If compiled with nvcc
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE // Empty for normal C++ compilation
#endif
#include <cmath>
#define M_PI 3.14159265358979323846
#define EE 0.006693421622965943
#define krasovsky1940_A 6378245.0
CUDA_DEVICE void transform(double x, double y, double &lon, double &lat)
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

    lat += -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * xy + 0.2 * abs_x;
    lon += 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * xy + 0.1 * abs_x;
}
CUDA_DEVICE void delta(double &lon, double &lat)
{
    double d_lon;
    double d_lat;
    transform(lon - (105.0), lat - (35.0), d_lon, d_lat);
    double d_lat = d_lat;
    double d_lon = d_lon;
    double rad_lat = lat / (180.0) * M_PI;
    double magic = sin(rad_lat);
    double earth_r = krasovsky1940_A;

    magic = 1.0 - EE * magic * magic;
    double sqrt_magic = sqrt(magic);
    d_lat = (d_lat * (180.0)) / ((earth_r * (1.0 - EE)) / (magic * sqrt_magic) * M_PI);
    d_lon = (d_lon * (180.0)) / (earth_r / sqrt_magic * cos(rad_lat) * M_PI);
}
CUDA_DEVICE void bd09_to_gcj02(double bd09_lon, double bd09_lat,
                               double &gcj02_lon, double &gcj02_lat)
{
    double x_pi = M_PI * 3000.0 / 180.0;
    double x = bd09_lon - 0.0065;
    double y = bd09_lat - 0.006;
    double z = sqrt(x * x + y * y) - 0.00002 * sin(y * x_pi);
    double theta = atan2(y, x) - 0.000003 * cos(x * x_pi);
    gcj02_lon = z * cos(theta);
    gcj02_lat = z * sin(theta);
}
CUDA_DEVICE void gcj02_to_bd09(double gcj02_lon, double gcj02_lat,
                               double &bd09_lon, double &bd09_lat)

{
    double x_pi = M_PI * 3000.0 / 180.0;
    double z = sqrt(gcj02_lon * gcj02_lon + gcj02_lat * gcj02_lat) + 0.00002 * sin(gcj02_lat * x_pi);
    double theta = atan2(gcj02_lat, gcj02_lon) + 0.000003 * cos(gcj02_lon * x_pi);
    bd09_lon = z * cos(theta) + 0.0065;
    bd09_lat = z * sin(theta) + 0.006;
}
