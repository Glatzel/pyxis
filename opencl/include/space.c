#ifndef PYXIS_SPACE_H
#define PYXIS_SPACE_H

static void cylindrical_to_cartesian_float(
    const float r,
    const float theta,
    const float z,
    float *out_x,
    float *out_y,
    float *out_z)
{
  *out_x = r * cos(theta);
  *out_y = r * sin(theta);
  *out_z = z;
}
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
void cartesian_to_cylindrical_f(
    const float x,
    const float y,
    const float z)
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
}
void to_cylindrical_f(const float x, const float y, const float z)
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

  return (sqrt(x * x + y * y), theta, z);
}
void to_polar_float(const float x, const float y)
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

  return (sqrt(x * x + y * y), theta);
}
void from_polar_float(const float r, const float theta, float *out_x, float *out_y)
{
  *out_x = r * cos(theta);
  *out_y = r * sin(theta);
}
#endif