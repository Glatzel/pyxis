mod context;
mod options;
mod output_fn;
mod record;
use bpaf::Bpaf;
use context::ContextTransform;
pub use options::*;
use pyxis::crypto::CryptoSpace;
use record::Record;
#[derive(Bpaf, Clone, Debug)]
pub enum TransformCommands {
    #[bpaf(command, adjacent)]
    /// Crypto coordinates between `BD09`, `GCJ02` and `WGS84`.
    Crypto {
        #[bpaf(short, long)]
        from: CryptoSpace,
        #[bpaf(short, long)]
        to: CryptoSpace,
    },

    #[bpaf(command, adjacent)]
    /// Converts projected XY coordinates from the height compensation plane to
    /// the sea level plane.
    DatumCompense {
        #[bpaf(short, long)]
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

    #[bpaf(command, adjacent)]
    /// Migrate2d.
    Migrate2d {
        #[bpaf(short, long)]
        given: MigrateOption2d,
        #[bpaf(short, long)]
        another: MigrateOption2d,
        #[bpaf(long)]
        another_x: f64,
        #[bpaf(long)]
        another_y: f64,
        #[bpaf(short, long)]
        rotate: f64,
        #[bpaf(short, long)]
        unit: options::RotateUnit,
    },

    #[bpaf(command, adjacent)]
    /// Normalize.
    Normalize {},

    #[bpaf(command, adjacent)]
    /// Transform coordinate from one known coordinate reference systems to
    /// another.
    ///
    ///  The `from` and `to` can be:
    ///  - an "AUTHORITY:CODE", like "EPSG:25832".
    ///  - a PROJ string, like "+proj=longlat +datum=WGS84". When using that
    ///    syntax, the unit is expected to be degrees.
    ///  - the name of a CRS as found in the PROJ database, e.g "WGS84",
    ///    "NAD27", etc.
    Proj {
        #[bpaf(short, long, argument("PROJ"))]
        from: String,
        #[bpaf(short, long, argument("PROJ"))]
        to: String,
    },

    #[bpaf(command, adjacent)]
    /// Rotate Coordinate.
    Rotate {
        #[bpaf(short, long)]
        value: f64,
        #[bpaf(short, long)]
        plane: RotatePlane,
        #[bpaf(short, long)]
        unit: options::RotateUnit,
        #[bpaf(long, fallback(0.0))]
        ox: f64,
        #[bpaf(long, fallback(0.0))]
        oy: f64,
        #[bpaf(long, fallback(0.0))]
        oz: f64,
    },

    #[bpaf(command, adjacent)]
    /// Scale Coordinate.
    Scale {
        #[bpaf(long, fallback(1.0))]
        sx: f64,
        #[bpaf(long, fallback(1.0))]
        sy: f64,
        #[bpaf(long, fallback(1.0))]
        sz: f64,
        #[bpaf(long, fallback(0.0))]
        ox: f64,
        #[bpaf(long, fallback(0.0))]
        oy: f64,
        #[bpaf(long, fallback(0.0))]
        oz: f64,
    },

    #[bpaf(command, adjacent)]
    /// Transforms coordinates between Cartesian, cylindrical, and spherical
    /// coordinate systems.
    Space {
        #[bpaf(short, long)]
        from: CoordSpace,
        #[bpaf(short, long)]
        to: CoordSpace,
    },

    #[bpaf(command, adjacent)]
    /// Translate Coordinate.
    Translate {
        #[bpaf(short, long, fallback(0.0))]
        tx: f64,
        #[bpaf(short, long, fallback(0.0))]
        ty: f64,
        #[bpaf(short, long, fallback(0.0))]
        tz: f64,
    },
}
pub fn execute(
    name: &str,
    x: f64,
    y: f64,
    z: f64,
    output_format: OutputFormat,
    cmds: Vec<TransformCommands>,
) {
    let mut ctx = ContextTransform { x, y, z };
    let mut records: Vec<Record> = vec![Record {
        idx: 0,
        method: "input".to_string(),
        parameter: serde_json::json!({}),
        output_x: ctx.x,
        output_y: ctx.y,
        output_z: ctx.z,
        output_x_name: "x".to_string(),
        output_y_name: "y".to_string(),
        output_z_name: "z".to_string(),
    }];
    for (i, cmd) in cmds.iter().enumerate() {
        clerk::debug!("step: {i}");
        clerk::debug!("cmd: {cmd:?}");
        match cmd {
            TransformCommands::Crypto { from, to } => {
                ctx.crypto(*from, *to);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "crypto".to_string(),
                    parameter: serde_json::json!({
                        "from": from.to_string(),
                        "to": to.to_string()
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "longitude".to_string(),
                    output_y_name: "latitude".to_string(),
                    output_z_name: "elevation".to_string(),
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
                    parameter: serde_json::json!({
                        "hb": hb,
                        "r": r,
                        "x0":x0,
                        "y0":y0
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "elevation".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Migrate2d {
                given,
                another,
                another_x,
                another_y,
                rotate,
                unit,
            } => {
                ctx.migrate2d(*given, *another, *another_x, *another_y, *rotate, *unit);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "crypto".to_string(),
                    parameter: serde_json::json!({
                        "given": given.to_string(),
                        "another": another.to_string(),
                        "another_x":another_x.to_string(),
                        "another_y":another_y.to_string(),
                        "rotate":rotate.to_string(),
                        "rotate_unit": unit.to_string()
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Normalize {} => {
                ctx.normalize();
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "normalize".to_string(),
                    parameter: serde_json::json!({}),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Proj { from, to } => {
                ctx.proj(from.as_str(), to.as_str()).unwrap();
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "proj".to_string(),
                    parameter: serde_json::json!({
                        "from": from.to_string(),
                        "to": to.to_string()
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Rotate {
                value,
                plane,
                unit,
                ox,
                oy,
                oz,
            } => {
                ctx.rotate(*value, *plane, *unit, *ox, *oy, *oz);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "rotate".to_string(),
                    parameter: serde_json::json!({
                        "value": value,
                        "plane": plane.to_string(),
                        "unit": unit.to_string(),
                        "origin_x":ox,
                        "origin_y":oy,
                        "origin_z":oz
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Scale {
                sx,
                sy,
                sz,
                ox,
                oy,
                oz,
            } => {
                ctx.scale(*sx, *sy, *sz, *ox, *oy, *oz);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "scale".to_string(),
                    parameter: serde_json::json!({
                        "scale_x": sx,
                        "scale_y": sy,
                        "scale_z": sz,
                        "origin_x":ox,
                        "origin_y":oy,
                        "origin_z":oz
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Space { from, to } => {
                ctx.space(*from, *to);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "space".to_string(),
                    parameter: serde_json::json!({
                        "from": from.to_string(),
                        "to": to.to_string()
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "longitude".to_string(),
                    output_y_name: "latitude".to_string(),
                    output_z_name: "elevation".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Translate { tx, ty, tz } => {
                ctx.translate(*tx, *ty, *tz);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "scale".to_string(),
                    parameter: serde_json::json!({
                        "translate_x": tx,
                        "translate_y": ty,
                        "translate_z": tz
                    }),
                    output_x: ctx.x,
                    output_y: ctx.y,
                    output_z: ctx.z,
                    output_x_name: "x".to_string(),
                    output_y_name: "y".to_string(),
                    output_z_name: "z".to_string(),
                };
                records.push(record);
            }
        }
        clerk::debug!("context x: {}, y: {}, z: {}", ctx.x, ctx.y, ctx.z);
    }
    // output
    match output_format {
        OutputFormat::Simple => output_fn::output_simple(records.last().unwrap()),
        OutputFormat::Plain => output_fn::output_plain(name, &records),
        OutputFormat::Json => output_fn::output_json(name, &records),
    }
}
