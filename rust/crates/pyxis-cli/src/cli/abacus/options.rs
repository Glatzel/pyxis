use std::fmt;
use std::str::FromStr;

use bpaf::Bpaf;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub enum OutputFormat {
    Simple,
    Verbose,
    Json,
}
impl FromStr for OutputFormat {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(Self::Simple),
            "verbose" => Ok(Self::Verbose),
            "json" => Ok(Self::Json),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Verbose => write!(f, "verbose"),
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
pub enum RotatePlane {
    Xy,
    Zx,
    Yz,
}

impl FromStr for RotatePlane {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "xy" => Ok(Self::Xy),
            "zx" => Ok(Self::Zx),
            "yz" => Ok(Self::Yz),

            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for RotatePlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotatePlane::Xy => write!(f, "XY"),
            RotatePlane::Zx => write!(f, "XZ"),
            RotatePlane::Yz => write!(f, "YZ"),
        }
    }
}
#[derive(Debug, Clone, Copy, Bpaf)]
pub enum RotateUnit {
    Degrees,
    Radians,
}

impl FromStr for RotateUnit {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "degrees" => Ok(Self::Degrees),
            "radians" => Ok(Self::Radians),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for RotateUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Degrees => write!(f, "Angle"),
            Self::Radians => write!(f, "Radians"),
        }
    }
}
#[derive(Debug, Clone, Copy, Bpaf)]
pub enum MigrateOption2d {
    Absolute,
    Origin,
    Relative,
}

impl FromStr for MigrateOption2d {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "absolute" => Ok(Self::Absolute),
            "origin" => Ok(Self::Origin),
            "relative" => Ok(Self::Relative),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for MigrateOption2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absolute => write!(f, "Absolute"),
            Self::Origin => write!(f, "Origin"),
            Self::Relative => write!(f, "Relative"),
        }
    }
}
