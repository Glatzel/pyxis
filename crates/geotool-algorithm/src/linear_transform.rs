pub fn rotate_matrix_2d(radians: f64) -> [[f64; 2]; 2] {
    [
        [radians.cos(), radians.sin()],
        [-radians.sin(), radians.cos()],
    ]
}
pub fn rotate_2d(x: f64, y: f64, rotate_matrix: &[[f64; 2]; 2]) -> (f64, f64) {
    let out_x = rotate_matrix[0][0] * x + rotate_matrix[0][1] * y;
    let out_y = rotate_matrix[1][0] * x + rotate_matrix[1][1] * y;
    (out_x, out_y)
}
