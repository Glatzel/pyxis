/// Creates a 2D rotation matrix for a given angle in radians.
///
/// This function generates the 2x2 rotation matrix that can be used to rotate
/// points in 2D space by a given angle.
///
/// # Arguments
///
/// * `radians` - The angle to rotate the matrix by, in radians.
///
/// # Returns
///
/// A 2x2 rotation matrix as a 2D array:
/// $$
/// [
///     [cos(angle), -sin(angle)],
///     [sin(angle), cos(angle)]
/// ]
/// $$
///
/// # Example
///
/// ```rust
/// let radians = std::f64::consts::PI / 2.0; // 90 degrees in radians
/// let rotation_matrix = rotate_matrix_2d(radians);
///
/// // Print the rotation matrix
/// println!("{:?}", rotation_matrix);
/// ```
///
/// This will output:
/// ```text
/// [[6.123233995736766e-17, -1.0], [1.0, 6.123233995736766e-17]]
/// ```
pub fn rotate_matrix_2d(radians: f64) -> [[f64; 2]; 2] {
    [
        [radians.cos(), -radians.sin()],
        [radians.sin(), radians.cos()],
    ]
}
pub fn rotate_2d(x: f64, y: f64, rotate_matrix: &[[f64; 2]; 2]) -> (f64, f64) {
    let out_x = rotate_matrix[0][0] * x + rotate_matrix[0][1] * y;
    let out_y = rotate_matrix[1][0] * x + rotate_matrix[1][1] * y;
    (out_x, out_y)
}
