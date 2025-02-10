/// Represents an ellipsoid with semi-major axis `a`, semi-minor axis `b`, eccentricity `e`,
/// squared eccentricity `e²`, flattening `f`, and inverse flattening `1/f` in the `geotool_algorithm` crate.
///
/// # Examples
///
/// ```
/// use geotool_algorithm::Ellipsoid;
/// use float_cmp::approx_eq;
///
/// let semi_major_axis = 6378137.0;  // Semi-major axis in meters
/// let inverse_flattening = 298.257223563;  // Inverse flattening
/// let ellipsoid = Ellipsoid::from(semi_major_axis, inverse_flattening);
///
/// assert_eq!(ellipsoid.semi_major_axis(), 6378137.0);
/// assert_eq!(ellipsoid.inverse_flattening(), 298.257223563);
/// assert!(approx_eq!(f64, ellipsoid.eccentricity(), 0.081819190842622, epsilon = 1e-12));
/// assert!(approx_eq!(f64, ellipsoid.flattening(), 0.0033528106647474805, epsilon = 1e-12));
/// ```
pub struct Ellipsoid {
    semi_major_axis: f64,    // Semi-major axis `a`
    semi_minor_axis: f64,    // Semi-minor axis `b`
    eccentricity: f64,       // Eccentricity `e`
    eccentricity2: f64,      // Squared eccentricity `e²`
    flattening: f64,         // Flattening `f`
    inverse_flattening: f64, // Inverse flattening `1/f`
}

impl Ellipsoid {
    /// Creates a new `Ellipsoid` from semi-major axis (`a`) and inverse flattening (`1/f`).
    ///
    /// # Arguments
    ///
    /// * `semi_major_axis` - The semi-major axis (`a`).
    /// * `inverse_flattening` - The inverse flattening (`1/f`).
    ///
    /// # Returns
    ///
    /// * A new `Ellipsoid` instance with all calculated properties.
    ///
    /// # Examples
    ///
    /// ```
    /// use geotool_algorithm::Ellipsoid;
    /// use float_cmp::approx_eq;
    ///
    /// let semi_major_axis = 6378137.0;  // WGS84 semi-major axis in meters
    /// let inverse_flattening = 298.257223563;  // WGS84 inverse flattening
    ///
    /// let ellipsoid = Ellipsoid::from_semi_major_and_invf(semi_major_axis, inverse_flattening);
    ///
    /// assert_eq!(ellipsoid.semi_major_axis(), 6378137.0);
    /// assert_eq!(ellipsoid.inverse_flattening(), 298.257223563);
    /// assert!(approx_eq!(f64, ellipsoid.eccentricity(), 0.081819190842622, epsilon = 1e-12));
    /// assert!(approx_eq!(f64, ellipsoid.flattening(), 0.0033528106647474805, epsilon = 1e-12));
    /// ```
    pub fn from_semi_major_and_invf(semi_major_axis: f64, inverse_flattening: f64) -> Self {
        let flattening = 1.0 / inverse_flattening;
        let semi_minor_axis = semi_major_axis * (1.0 - flattening);
        let eccentricity2 = 2.0 * flattening - flattening * flattening;
        let eccentricity = eccentricity2.sqrt();

        Ellipsoid {
            semi_major_axis,
            semi_minor_axis,
            eccentricity,
            eccentricity2,
            flattening,
            inverse_flattening,
        }
    }

    /// Returns the semi-major axis of the ellipsoid.
    pub fn semi_major_axis(&self) -> f64 {
        self.semi_major_axis
    }

    /// Returns the semi-minor axis of the ellipsoid.
    pub fn semi_minor_axis(&self) -> f64 {
        self.semi_minor_axis
    }

    /// Returns the eccentricity (`e`) of the ellipsoid.
    pub fn eccentricity(&self) -> f64 {
        self.eccentricity
    }

    /// Returns the squared eccentricity (`e²`) of the ellipsoid.
    pub fn eccentricity2(&self) -> f64 {
        self.eccentricity2
    }

    /// Returns the flattening (`f`) of the ellipsoid.
    pub fn flattening(&self) -> f64 {
        self.flattening
    }

    /// Returns the inverse flattening (`1/f`) of the ellipsoid.
    pub fn inverse_flattening(&self) -> f64 {
        self.inverse_flattening
    }
}
