use bpaf::Bpaf;
use std::{fmt, str::FromStr};
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Simple,
    Plain,
    Json,
}
impl FromStr for OutputFormat {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(Self::Simple),
            "plain" => Ok(Self::Plain),
            "json" => Ok(Self::Json),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Plain => write!(f, "plain"),
            Self::Json => write!(f, "json"),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum CoordSpace {
    Cartesian,
    Cylindrical,
    Spherical,
}
impl FromStr for CoordSpace {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cartesian" => Ok(Self::Cartesian),
            "cylindrical" => Ok(Self::Cylindrical),
            "spherical" => Ok(Self::Spherical),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for CoordSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cartesian => write!(f, "Cartesian"),
            Self::Cylindrical => write!(f, "Cylindrical"),
            Self::Spherical => write!(f, "Spherical"),
        }
    }
}
#[derive(Debug, Clone, Copy, Bpaf)]
pub enum CryptoSpace {
    BD09,
    GCJ02,
    WGS84,
}

impl FromStr for CryptoSpace {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BD09" => Ok(Self::BD09),
            "GCJ02" => Ok(Self::GCJ02),
            "WGS84" => Ok(Self::WGS84),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for CryptoSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BD09 => write!(f, "BD09"),
            Self::GCJ02 => write!(f, "GCJ02"),
            Self::WGS84 => write!(f, "WGS84"),
        }
    }
}
#[derive(Debug, Clone, Copy, Bpaf)]
pub enum RotateOrder {
    Xyz,
    Xzy,
    Yzx,
    Yxz,
    Zxy,
    Zyx,
}

impl FromStr for RotateOrder {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "xyz" => Ok(Self::Xyz),
            "xzy" => Ok(Self::Xzy),
            "yzx" => Ok(Self::Yzx),
            "yxz" => Ok(Self::Yxz),
            "zxy" => Ok(Self::Zxy),
            "zyx" => Ok(Self::Zyx),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for RotateOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotateOrder::Xyz => write!(f, "Xyz"),
            RotateOrder::Xzy => write!(f, "Xzy"),
            RotateOrder::Yzx => write!(f, "Yzx"),
            RotateOrder::Yxz => write!(f, "Yxz"),
            RotateOrder::Zxy => write!(f, "Zxy"),
            RotateOrder::Zyx => write!(f, "Zyx"),
        }
    }
}
#[derive(Debug, Clone, Copy, Bpaf)]
pub enum RotateUnit {
    Angle,
    Radians,
}

impl FromStr for RotateUnit {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "angle" => Ok(Self::Angle),
            "radians" => Ok(Self::Radians),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for RotateUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Angle => write!(f, "Angle"),
            Self::Radians => write!(f, "Radians"),
        }
    }
}
