crate::create_readonly_struct!(
    Factors,
    "Various cartographic properties, such as scale factors, angular distortion and meridian convergence. Calculated with [`crate::Proj::factors()`]."
    "# References"
    "- * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_FACTORS>",
    {meridional_scale: f64,r"Meridional scale at coordinate $`(\lambda, \phi)`$."},
    {parallel_scale: f64,r"Parallel scale at coordinate $`(\lambda, \phi)`$."},
    {areal_scale: f64,r"Areal scale factor at coordinate $`(\lambda, \phi)`$."},

    {angular_distortion: f64,r"Angular distortion at coordinate $`(\lambda, \phi)`$."},
    {meridian_parallel_angle: f64,r"Meridian/parallel angle, $\theta'$ , at coordinate $`(\lambda, \phi)`$."},
    {meridian_convergence: f64,r"Meridian convergence at coordinate $`(\lambda, \phi)`$. Sometimes also described as grid declination."},

    {tissot_semimajor: f64,"Maximum scale factor."},
    {tissot_semiminor: f64,"Minimum scale factor."},

    {dx_dlam: f64,r"Partial derivative $`\cfrac{\partial x}{\partial \lambda}`$ of coordinate $`(\lambda, \phi)`$."},
    {dx_dphi: f64,r"Partial derivative $`\cfrac{\partial y}{\partial \lambda}`$ of coordinate $`(\lambda, \phi)`$."},
    {dy_dlam: f64,r"Partial derivative $`\cfrac{\partial x}{\partial \phi}`$ of coordinate $`(\lambda, \phi)`$."},
    {dy_dphi: f64,r"Partial derivative $`\cfrac{\partial y}{\partial \phi}`$ of coordinate $`(\lambda, \phi)`$."}
);
