crate::create_readonly_struct!(
    PjFactors,
    "https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_FACTORS",
    {meridional_scale: f64},
    {parallel_scale: f64},
    {areal_scale: f64},

    {angular_distortion: f64},
    {meridian_parallel_angle: f64},
    {meridian_convergence: f64},

    {tissot_semimajor: f64},
    {tissot_semiminor: f64},

    {dx_dlam: f64},
    {dx_dphi: f64},
    {dy_dlam: f64},
    {dy_dphi: f64}
);
