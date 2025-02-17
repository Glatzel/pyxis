pub fn rel_2d(
    origin_x: f64,
    origin_y: f64,
    abs_x: f64,
    abs_y: f64,
    rotate_matrix: &[[f64; 2]; 2],
) -> (f64, f64) {
    let temp_x = abs_x - origin_x;
    let temp_y = abs_y - origin_y;
    (
        rotate_matrix[0][0] * temp_x - rotate_matrix[0][1] * temp_y,
        -rotate_matrix[1][0] * temp_x + rotate_matrix[1][1] * temp_y,
    )
}
pub fn abs_2d(
    origin_x: f64,
    origin_y: f64,
    rel_x: f64,
    rel_y: f64,
    rotate_matrix: &[[f64; 2]; 2],
) -> (f64, f64) {
    (
        -rotate_matrix[0][0] * rel_x + rotate_matrix[0][1] * rel_x + origin_x,
        rotate_matrix[1][0] * rel_y - rotate_matrix[1][1] * rel_y + origin_y,
    )
}
pub fn origin_2d(
    abs_x: f64,
    abs_y: f64,
    rel_x: f64,
    rel_y: f64,
    rotate_matrix: &[[f64; 2]; 2],
) -> (f64, f64) {
    (
        -rotate_matrix[0][0] * rel_x + rotate_matrix[0][1] * rel_x + abs_x,
        rotate_matrix[1][0] * rel_y - rotate_matrix[1][1] * rel_y + abs_y,
    )
}

// pub fn rotate_matrix_3d(rx: f64, ry: f64, rz: f64, order: RotareOrder) -> [[f64; 3]; 3] {
//     [
//         [radians.cos(), radians.sin()],
//         [-radians.sin(), radians.cos()],
//     ]
// }
