use assert_cmd::Command;
use assert_cmd::assert::Assert;

fn print_output(cmd: &Assert) {
    let output = cmd.get_output();
    let stdout = String::from_utf8_lossy(output.stdout.as_slice());
    let stderr = String::from_utf8_lossy(output.stderr.as_slice());

    // Print results
    if output.status.success() {
        println!("Output:\n{stdout}");
    } else {
        eprintln!("Error:\n{stderr}");
    }
}
#[test]
fn test_jiaxing_bump_station() -> mischief::Result<()> {
    let cmd = Command::cargo_bin("pyxis")?
        .args([
            "abacus",
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
        .assert()
        .success();
    print_output(&cmd);
    Ok(())
}
#[test]
fn test_zhengyong_expressway_dehua_east_interchange() -> mischief::Result<()> {
    let cmd = Command::cargo_bin("pyxis")
        ?
    .args([
            "-v",
            "abacus",
            "-n", "Zhengyong expressway Dehua east interchange",
            "-x", "469704.6693",
            "-y", "2821940.796",
            "-z", "0",
            "-o", "verbose",
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
        .assert()
        .success();
    print_output(&cmd);
    Ok(())
}
