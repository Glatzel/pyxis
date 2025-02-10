/// Represents an ellipsoid with semi-major axis `a` and semi-minor axis `b`.
/// Provides methods to compute its eccentricity, squared eccentricity, flattening, and inverse flattening.
pub struct Ellipsoid {
    a: f64, // Semi-major axis
    b: f64, // Semi-minor axis
}

impl Ellipsoid {
    /// Creates a new `Ellipsoid` with the given semi-major and semi-minor axes.
    ///
    /// # Parameters
    ///
    /// - `a`: The semi-major axis.
    /// - `b`: The semi-minor axis.
    ///
    /// # Returns
    ///
    /// - A new `Ellipsoid` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// let ellipsoid = geotool_algorithm::Ellipsoid::new(6378137.0, 6356752.314245);
    /// assert_eq!(ellipsoid.a(), 6378137.0);
    /// assert_eq!(ellipsoid.b(), 6356752.314245);
    /// ```
    pub fn new(a: f64, b: f64) -> Self {
        Ellipsoid { a, b }
    }

    /// Calculates the eccentricity (`e`) of the ellipsoid.
    ///
    /// # Returns
    ///
    /// - The eccentricity `e` of the ellipsoid.
    ///
    /// # Examples
    ///
    /// ```
    /// let ellipsoid = geotool_algorithm::Ellipsoid::new(6378137.0, 6356752.314245);
    /// assert!((ellipsoid.eccentricity() - 0.081819190842622).abs() < 1e-12);
    /// ```
    pub fn eccentricity(&self) -> f64 {
        (1.0 - (self.b * self.b) / (self.a * self.a)).sqrt()
    }

    /// Calculates the squared eccentricity (`e²`) of the ellipsoid.
    ///
    /// # Returns
    ///
    /// * The squared eccentricity `e²` of the ellipsoid.
    ///
    /// # Examples
    ///
    /// ```
    /// let ellipsoid = geotool_algorithm::Ellipsoid::new(6378137.0, 6356752.314245);
    /// assert!((ellipsoid.squared_eccentricity() - 0.006694379990141).abs() < 1e-12);
    /// ```
    pub fn squared_eccentricity(&self) -> f64 {
        1.0 - (self.b * self.b) / (self.a * self.a)
    }

    /// Calculates the flattening (`f`) of the ellipsoid.
    ///
    /// # Returns
    ///
    /// - The flattening `f` of the ellipsoid.
    ///
    /// # Examples
    ///
    /// ```
    /// let ellipsoid = geotool_algorithm::Ellipsoid::new(6378137.0, 6356752.314245);
    /// assert!((ellipsoid.flattening() - 0.0033528106647474805).abs() < 1e-12);
    /// ```
    pub fn flattening(&self) -> f64 {
        (self.a - self.b) / self.a
    }

    /// Calculates the inverse flattening (`1/f`) of the ellipsoid.
    ///
    /// # Returns
    ///
    /// - The inverse flattening `1/f` of the ellipsoid.
    ///
    /// # Examples
    ///
    /// ```
    /// let ellipsoid = geotool_algorithm::Ellipsoid::new(6378137.0, 6356752.314245);
    /// assert_eq!(ellipsoid.inverse_flattening(), 298.257223563);
    /// ```
    pub fn inverse_flattening(&self) -> f64 {
        self.a / (self.a - self.b)
    }
}
