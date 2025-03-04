/// Converts a `f64` angle value (in decimal degrees) to degrees, minutes, and seconds (DMS).
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let angle: f64 = 30.76;
/// let dms = pyxis::angle_to_dms(angle);
/// assert_eq!(dms.0, 30);
/// assert_eq!(dms.1, 45);
/// assert_approx_eq!(f64, dms.2, 36.0, epsilon = 1e-6);
/// ```
pub fn angle_to_dms(angle: f64) -> (i32, i32, f64) {
    let degree = angle.trunc() as i32; // Get degrees
    let minutes_float = (angle - degree as f64) * 60.0; // Get fractional part and convert to minutes
    let minute = minutes_float.trunc() as i32; // Get minutes as integer
    let second = (minutes_float - minute as f64) * 60.0; // Get seconds
    (degree, minute, second)
}

/// Converts the DMS (degrees, minutes, seconds) format back into a decimal degree value.
///
/// # Returns
///
/// A `f64` representing the angle in decimal degrees.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let angle = pyxis::dms_to_angle(30,45,36.0);
/// assert_approx_eq!(f64, angle, 30.76, epsilon = 1e-6);
/// ```
pub fn dms_to_angle(degree: i32, minute: i32, second: f64) -> f64 {
    degree as f64 + minute as f64 / 60.0 + second / 3600.0
}
