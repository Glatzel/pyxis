use crate::GeoFloat;

/// Represents an ellipsoid with semi-major axis `a`, semi-minor axis `b`,
/// eccentricity `e`, squared eccentricity `e²`, flattening `f`, and inverse
/// flattening `1/f` in the `pyxis` crate.
///
/// # Examples
///
/// ```
/// use pyxis::Ellipsoid;
/// use float_cmp::assert_approx_eq;
///
/// let semi_major_axis = 6378137.0;  // Semi-major axis in meters
/// let inverse_flattening = 298.257223563;  // Inverse flattening
/// let ellipsoid = Ellipsoid::from_semi_major_and_invf(semi_major_axis, inverse_flattening);
///
/// assert_eq!(ellipsoid.semi_major_axis(), 6378137.0);
/// assert_eq!(ellipsoid.inverse_flattening(), 298.257223563);
/// assert_approx_eq!(f64, ellipsoid.eccentricity(), 0.081819190842622, epsilon = 1e-12);
/// assert_approx_eq!(f64, ellipsoid.flattening(), 0.0033528106647474805, epsilon = 1e-12);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ellipsoid<T> {
    semi_major_axis: T,    // Semi-major axis `a`
    semi_minor_axis: T,    // Semi-minor axis `b`
    eccentricity: T,       // Eccentricity `e`
    eccentricity2: T,      // Squared eccentricity `e²`
    flattening: T,         // Flattening `f`
    inverse_flattening: T, // Inverse flattening `1/f`
}

impl<T: GeoFloat> Ellipsoid<T> {
    /// Creates a new `Ellipsoid` from semi-major axis (`a`) and inverse
    /// flattening (`1/f`).
    ///
    /// # Arguments
    ///
    /// - `semi_major_axis` - The semi-major axis (`a`).
    /// - `inverse_flattening` - The inverse flattening (`1/f`).
    ///
    /// # Returns
    ///
    /// - A new `Ellipsoid` instance with all calculated properties.
    ///
    /// # Examples
    ///
    /// ```
    /// use pyxis::Ellipsoid;
    /// use float_cmp::assert_approx_eq;
    ///
    /// let semi_major_axis = 6378137.0;  // WGS84 semi-major axis in meters
    /// let inverse_flattening = 298.257223563;  // WGS84 inverse flattening
    ///
    /// let ellipsoid = Ellipsoid::from_semi_major_and_invf(semi_major_axis, inverse_flattening);
    ///
    /// assert_eq!(ellipsoid.semi_major_axis(), 6378137.0);
    /// assert_eq!(ellipsoid.inverse_flattening(), 298.257223563);
    /// assert_approx_eq!(f64, ellipsoid.eccentricity(), 0.081819190842622, epsilon = 1e-12);
    /// assert_approx_eq!(f64, ellipsoid.flattening(), 0.0033528106647474805, epsilon = 1e-12);
    /// ```
    pub fn from_semi_major_and_invf(semi_major_axis: T, inverse_flattening: T) -> Self {
        let flattening: T = T::ONE / inverse_flattening;
        let semi_minor_axis = semi_major_axis * (T::ONE - flattening);
        let eccentricity2: T = T::TWO * flattening - flattening * flattening;
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
    /// Create an Ellipsoid from the semi-major axis and semi-minor axis.
    ///
    /// # Arguments
    ///
    /// - `semi_major_axis`: The semi-major axis (`a`) of the ellipsoid (in
    ///   meters).
    /// - `semi_minor_axis`: The semi-minor axis (`b`) of the ellipsoid (in
    ///   meters).
    ///
    /// # Returns
    ///
    /// Returns an `Ellipsoid` instance with the provided semi-major axis and
    /// semi-minor axis, along with the calculated values for flattening,
    /// eccentricity, and inverse flattening.
    ///
    /// # Calculation
    ///
    /// The function computes:
    /// - Flattening `f = (a - b) / a`
    /// - Inverse flattening `1/f`
    /// - Eccentricity squared `e² = 2f - f²`
    /// - Eccentricity `e = sqrt(e²)`
    ///
    /// # Example
    ///
    /// Create an `Ellipsoid` from semi-major axis and semi-minor axis:
    ///
    /// ```
    /// use pyxis::Ellipsoid;
    /// let ellipsoid = Ellipsoid::from_semi_axis(6378137.0, 6356752.314245);
    /// assert_eq!(ellipsoid.semi_major_axis(), 6378137.0);
    /// assert_eq!(ellipsoid.semi_minor_axis(), 6356752.314245);
    /// float_cmp::assert_approx_eq!(f64, ellipsoid.eccentricity(), 0.081819190842622, epsilon = 1e-6);
    /// float_cmp::assert_approx_eq!(f64, ellipsoid.flattening(), 0.0033528106647474805, epsilon = 1e-10);
    /// ```
    ///
    /// # Notes
    /// This function is useful when you only know the semi-major axis and
    /// semi-minor axis and need to compute the rest of the ellipsoid
    /// parameters.
    pub fn from_semi_axis(semi_major_axis: T, semi_minor_axis: T) -> Self {
        // Calculate flattening
        let flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis;
        // Calculate inverse flattening
        let inverse_flattening: T = T::ONE / flattening;
        // Calculate eccentricity²
        let eccentricity2: T = T::TWO * flattening - flattening.powi(2);
        // Calculate eccentricity
        let eccentricity = eccentricity2.sqrt();

        Ellipsoid {
            semi_major_axis,
            semi_minor_axis,
            flattening,
            inverse_flattening,
            eccentricity,
            eccentricity2,
        }
    }
    /// Returns the semi-major axis of the ellipsoid.
    pub fn semi_major_axis(&self) -> T { self.semi_major_axis }

    /// Returns the semi-minor axis of the ellipsoid.
    pub fn semi_minor_axis(&self) -> T { self.semi_minor_axis }

    /// Returns the eccentricity (`e`) of the ellipsoid.
    pub fn eccentricity(&self) -> T { self.eccentricity }

    /// Returns the squared eccentricity (`e²`) of the ellipsoid.
    pub fn eccentricity2(&self) -> T { self.eccentricity2 }

    /// Returns the flattening (`f`) of the ellipsoid.
    pub fn flattening(&self) -> T { self.flattening }

    /// Returns the inverse flattening (`1/f`) of the ellipsoid.
    pub fn inverse_flattening(&self) -> T { self.inverse_flattening }
}
