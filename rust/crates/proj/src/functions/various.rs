impl crate::Pj {
    fn _roundtrip(&self) { unimplemented!() }
    fn _factors(&self) { unimplemented!() }
    fn _angular_input(&self) { unimplemented!() }
    fn _angular_output(&self) { unimplemented!() }
    fn _degree_input(&self) { unimplemented!() }
    fn _degree_output(&self) { unimplemented!() }
}

#[cfg(any(feature = "unrecommended", test))]
pub fn coord(x: f64, y: f64, z: f64, t: f64) -> proj_sys::PJ_COORD {
    unsafe { proj_sys::proj_coord(x, y, z, t) }
}
#[deprecated(note = "Use `f64::to_radians(self)` instead")]
fn _torad() { unimplemented!() }
#[deprecated(note = "Use `f64::to_degrees(self)` instead")]
fn _todeg() { unimplemented!() }
pub fn dmstor() { unimplemented!() }
pub fn rtodms() { unimplemented!() }
pub fn rtodms2() { unimplemented!() }
