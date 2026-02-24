use derive_getters::Getters;

///Various cartographic properties, such as scale factors, angular distortion
/// and meridian convergence. Calculated with [`crate::Proj::factors()`]."]
///
///# References
///
/// - * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_FACTORS>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct Factors {
    ///Meridional scale at coordinate $`(\lambda, \phi)`$.
    meridional_scale: f64,
    ///Parallel scale at coordinate $`(\lambda, \phi)`$.
    parallel_scale: f64,
    ///Areal scale factor at coordinate $`(\lambda, \phi)`$.
    areal_scale: f64,
    ///Angular distortion at coordinate $`(\lambda, \phi)`$.
    angular_distortion: f64,
    ///Meridian/parallel angle, $\theta'$ , at coordinate $`(\lambda, \phi)`$.
    meridian_parallel_angle: f64,
    ///Meridian convergence at coordinate $`(\lambda, \phi)`$. Sometimes also described as grid declination.
    meridian_convergence: f64,
    ///Maximum scale factor.
    tissot_semimajor: f64,
    ///Minimum scale factor.
    tissot_semiminor: f64,
    ///Partial derivative $`\cfrac{\partial x}{\partial \lambda}`$ of coordinate $`(\lambda, \phi)`$.
    dx_dlam: f64,
    ///Partial derivative $`\cfrac{\partial y}{\partial \lambda}`$ of coordinate $`(\lambda, \phi)`$.
    dx_dphi: f64,
    ///Partial derivative $`\cfrac{\partial x}{\partial \phi}`$ of coordinate $`(\lambda, \phi)`$.
    dy_dlam: f64,
    ///Partial derivative $`\cfrac{\partial y}{\partial \phi}`$ of coordinate $`(\lambda, \phi)`$.
    dy_dphi: f64,
}
impl Factors {
    pub fn new(
        meridional_scale: f64,
        parallel_scale: f64,
        areal_scale: f64,
        angular_distortion: f64,
        meridian_parallel_angle: f64,
        meridian_convergence: f64,
        tissot_semimajor: f64,
        tissot_semiminor: f64,
        dx_dlam: f64,
        dx_dphi: f64,
        dy_dlam: f64,
        dy_dphi: f64,
    ) -> Self {
        Factors {
            meridional_scale,
            parallel_scale,
            areal_scale,
            angular_distortion,
            meridian_parallel_angle,
            meridian_convergence,
            tissot_semimajor,
            tissot_semiminor,
            dx_dlam,
            dx_dphi,
            dy_dlam,
            dy_dphi,
        }
    }
}
