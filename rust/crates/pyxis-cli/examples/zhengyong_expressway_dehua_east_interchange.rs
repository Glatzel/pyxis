use std::process::Command;

use miette::IntoDiagnostic;

fn main() -> miette::Result<()> {
    let output = Command::new("pyxis")
        .args([
            "-v",
            "transform",
            "-n", "Zhengyong expressway Dehua east interchange",
            "-x", "469704.6693",
            "-y", "2821940.796",
            "-z", "0",
            "-o", "plain",
        ])
        .args([
            "datum-compensate",
            "--hb", "400",
            "-r", "6378137",
            "--x0", "500000",
            "--y0", "0",
        ])
        .args([
            "proj",
            "--from", "+proj=tmerc +lat_0=0 +lon_0=118.5 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs",
            "--to", "+proj=longlat +datum=WGS84 +no_defs +type=crs",
        ])
        .output()
        .into_diagnostic()
       ?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Print results
    if output.status.success() {
        println!("Output:\n{}", stdout);
    } else {
        eprintln!("Error:\n{}", stderr);
    }
    Ok(())
}
