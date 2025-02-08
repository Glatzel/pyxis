mod options;
use bpaf::Bpaf;
pub use options::*;
mod context;
use context::ContextTransform;
mod record;
use record::Record;
mod output_fn;
#[derive(Bpaf, Clone, Debug)]
pub enum TransformCommands {
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Crypto coordinates between `BD09`, `GCJ02` and `WGS84`.
    Crypto {
        #[bpaf(short, long)]
        from: CryptoSpace,
        #[bpaf(short, long)]
        to: CryptoSpace,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Converts projected XY coordinates from the height compensation plane to the sea level plane.
    DatumCompense {
        #[bpaf(long)]
        /// Elevation of the height compensation plane (in meters).
        hb: f64,
        #[bpaf(short, long)]
        /// Radius of the Earth (in meters).
        radius: f64,
        #[bpaf(long)]
        /// X coordinate system origin (in meters).
        x0: f64,
        #[bpaf(long)]
        /// Y coordinate system origin (in meters).
        y0: f64,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Converts geodetic coordinates (longitude/L, latitude/B, height/H) to Cartesian coordinates (X, Y, Z).
    Lbh2xyz {
        #[bpaf(short('a'), long)]
        /// Semimajor radius of the ellipsoid axis
        major_radius: f64,
        #[bpaf(long("invf"))]
        /// Inverse flattening of the ellipsoid.
        inverse_flattening: f64,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Transform coordinate from one known coordinate reference systems to another.
    ///
    ///  The `from` and `to` can be:
    ///  - an "AUTHORITY:CODE", like "EPSG:25832".
    ///  - a PROJ string, like "+proj=longlat +datum=WGS84". When using that syntax, the unit is expected to be degrees.
    ///  - the name of a CRS as found in the PROJ database, e.g "WGS84", "NAD27", etc.
    Proj {
        #[bpaf(short, long, argument("PROJ"))]
        from: String,
        #[bpaf(short, long, argument("PROJ"))]
        to: String,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Converts Cartesian coordinates (X, Y, Z) to geodetic coordinates (Longitude, Latitude, Height).
    Xyz2lbh {
        #[bpaf(short('a'), long)]
        /// Semimajor radius of the ellipsoid axis
        major_radius: f64,
        #[bpaf(long("invf"))]
        /// Inverse flattening of the ellipsoid.
        inverse_flattening: f64,
    },
}
pub fn execute(x: f64, y: f64, z: f64, output_format: OutputFormat, cmds: Vec<TransformCommands>) {
    let mut ctx = ContextTransform { x, y, z };
    let mut records: Vec<Record> = vec![Record {
        idx: 0,
        method: "input".to_string(),
        from: "".to_string(),
        to: "".to_string(),
        ox: ctx.x,
        oy: ctx.y,
        oz: ctx.z,
        ox_name: "x".to_string(),
        oy_name: "y".to_string(),
        oz_name: "z".to_string(),
    }];
    for (i, cmd) in cmds.iter().enumerate() {
        match cmd {
            TransformCommands::Crypto { from, to } => {
                ctx.crypto(*from, *to);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "crypto".to_string(),
                    from: from.to_string(),
                    to: to.to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "longitude".to_string(),
                    oy_name: "latitude".to_string(),
                    oz_name: "elevation".to_string(),
                };
                records.push(record);
            }
            TransformCommands::DatumCompense {
                hb,
                radius: r,
                x0,
                y0,
            } => {
                ctx.datum_compense(*hb, *r, *x0, *y0);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "datum compense".to_string(),
                    from: "".to_string(),
                    to: "".to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "x".to_string(),
                    oy_name: "y".to_string(),
                    oz_name: "elevation".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Lbh2xyz {
                major_radius: semi_major_axis,
                inverse_flattening,
            } => {
                ctx.lbh2xyz(*semi_major_axis, *inverse_flattening);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "lbh2xyz".to_string(),
                    from: "lbh".to_string(),
                    to: "xyz".to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "x".to_string(),
                    oy_name: "y".to_string(),
                    oz_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Proj { from, to } => {
                ctx.cvt_proj(from.as_str(), to.as_str()).unwrap();
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "proj".to_string(),
                    from: from.to_string(),
                    to: to.to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "x".to_string(),
                    oy_name: "y".to_string(),
                    oz_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Xyz2lbh {
                major_radius: semi_major_axis,
                inverse_flattening,
            } => {
                ctx.xyz2lbh(*semi_major_axis, *inverse_flattening, None, None);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "xyz2lbh".to_string(),
                    from: "xyz".to_string(),
                    to: "lbh".to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "longitude".to_string(),
                    oy_name: "latitude".to_string(),
                    oz_name: "elevation".to_string(),
                };
                records.push(record);
            }
        }
    }
    // output
    match output_format {
        OutputFormat::Simple => output_fn::output_simple(records.last().unwrap()),
        OutputFormat::Plain => output_fn::output_plain(&records),
        OutputFormat::Json => output_fn::output_json(&records),
    }
}
