///https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INFO
pub struct PjInfo {
    major: i32,
    minor: i32,
    patch: i32,
    release: String,
    version: String,
    searchpath: String,
}
impl PjInfo {
    pub fn new(
        major: i32,
        minor: i32,
        patch: i32,
        release: String,
        version: String,
        searchpath: String,
    ) -> Self {
        Self {
            major,
            minor,
            patch,
            release,
            version,
            searchpath,
        }
    }
    pub fn major(&self) -> i32 {
        self.major
    }
    pub fn minor(&self) -> i32 {
        self.minor
    }
    pub fn patch(&self) -> i32 {
        self.patch
    }
    pub fn release(&self) -> &str {
        &self.release
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn searchpath(&self) -> &str {
        &self.searchpath
    }
}
pub struct PjProjInfo {
    id: String,
    description: String,
    definition: String,
    has_inverse: bool,
    accuracy: f64,
}
impl PjProjInfo {
    pub(crate) fn new(
        id: String,
        description: String,
        definition: String,
        has_inverse: bool,
        accuracy: f64,
    ) -> Self {
        Self {
            id,
            description,
            definition,
            has_inverse,
            accuracy,
        }
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn description(&self) -> &str {
        &self.description
    }
    pub fn definition(&self) -> &str {
        &self.definition
    }
    pub fn has_inverse(&self) -> bool {
        self.has_inverse
    }
    pub fn accuracy(&self) -> f64 {
        self.accuracy
    }
}
pub struct PjGridInfo {
    gridname: String,
    filename: String,
    format: String,
    // lowerleft: String,
    // upperright: String,
    n_lon: i32,
    n_lat: i32,
    cs_lon: f64,
    cs_lat: f64,
}
impl PjGridInfo {
    pub(crate) fn new() -> Self {
        unimplemented!()
    }
    pub fn gridname(&self) -> &str {
        &self.gridname
    }
    pub fn filename(&self) -> &str {
        &self.filename
    }
    pub fn format(&self) -> &str {
        &self.format
    }
    pub fn lowerleft(&self) -> &str {
        unimplemented!()
    }
    pub fn upperright(&self) -> &str {
        unimplemented!()
    }
    pub fn n_lon(&self) -> i32 {
        self.n_lon
    }
    pub fn n_lat(&self) -> i32 {
        self.n_lat
    }
    pub fn cs_lon(&self) -> f64 {
        self.cs_lon
    }
    pub fn cs_lat(&self) -> f64 {
        self.cs_lat
    }
}
pub struct PjInitInfo {
    name: String,
    filename: String,
    version: String,
    origin: String,
    lastupdate: String,
}
impl PjInitInfo {
    pub(crate) fn new(
        name: String,
        filename: String,
        version: String,
        origin: String,
        lastupdate: String,
    ) -> Self {
        Self {
            name,
            filename,
            version,
            origin,
            lastupdate,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn filename(&self) -> &str {
        &self.filename
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn lastupdate(&self) -> &str {
        &self.lastupdate
    }
}
