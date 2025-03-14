#ifndef PYXIS_SPACE_H
#define PYXIS_SPACE_H

void cartesian_to_spherical_float(
    const float x,
    const float y,
    const float z,
    float *out_u,
    float *out_v,
    float *out_r)
{
  float r = sqrt(x * x + y * y + z * z);
  float u = atan2(y, x) + M_PI;
  float v = acos(z / r);
  *out_u = r;
  *out_v = u;
  *out_r = v;
}
void spherical_to_cartesian_float(
    const float u,
    const float v,
    const float r,
    float *out_x,
    float *out_y,
    float *out_z)
{
  *out_x = sin(v) * cos(u) * r;
  *out_y = sin(v) * sin(u) * r;
  *out_z = cos(v) * r;
}

void cartesian_to_polar_float(const float x, const float y, float *out_r, float *out_theta)
{
  float theta;
  if (fabs(x) + fabs(y) < 0.000001)
  {
    theta = 0.0;
  }
  else
  {
    theta = atan2(y, x);
    theta = theta >= 0 ? theta : theta + 2 * M_PI;
  }

  *out_r = sqrt(x * x + y * y);
  *out_theta = theta;
}
void polar_to_cartesian_float(const float r, const float theta, float *out_x, float *out_y)
{
  *out_x = r * cos(theta);
  *out_y = r * sin(theta);
}
#endif
