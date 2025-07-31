use std::process::Command;

use miette::IntoDiagnostic;
fn main() -> miette::Result<()> {
    let output = Command::new("pyxis")
        .args([
            "transform",
            "-n",
            "Jiaxing bump station",
            "-x",
            "121.091701",
            "-y",
            "30.610765",
            "-z",
            "0",
            "-o",
            "json",
        ])
        .args(["crypto", "--from", "wgs84", "--to", "gcj02"])
        .output()
        .into_diagnostic()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Print results
    if output.status.success() {
        println!("Output:\n{stdout}");
    } else {
        eprintln!("Error:\n{stderr}");
    }
    Ok(())
}
