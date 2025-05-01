pub struct PjFactors {
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
}
impl PjFactors {
    pub(crate) fn new(
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
        Self {
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
    pub fn meridional_scale(&self) -> f64 {
        self.meridional_scale
    }
    pub fn parallel_scale(&self) -> f64 {
        self.parallel_scale
    }
    pub fn areal_scale(&self) -> f64 {
        self.areal_scale
    }
    pub fn angular_distortion(&self) -> f64 {
        self.angular_distortion
    }
    pub fn meridian_parallel_angle(&self) -> f64 {
        self.meridian_parallel_angle
    }
    pub fn meridian_convergence(&self) -> f64 {
        self.meridian_convergence
    }
    pub fn tissot_semimajor(&self) -> f64 {
        self.tissot_semimajor
    }
    pub fn tissot_semiminor(&self) -> f64 {
        self.tissot_semiminor
    }
    pub fn dx_dlam(&self) -> f64 {
        self.dx_dlam
    }
    pub fn dx_dphi(&self) -> f64 {
        self.dx_dphi
    }
    pub fn dy_dlam(&self) -> f64 {
        self.dy_dlam
    }
    pub fn dy_dphi(&self) -> f64 {
        self.dy_dphi
    }
}
