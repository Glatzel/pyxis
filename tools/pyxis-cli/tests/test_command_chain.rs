use assert_cmd::Command;
#[test]
fn test_jiaxing_bump_station() -> mischief::Result<()> {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
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
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
    Ok(())
}
#[test]
fn test_zhengyong_expressway_dehua_east_interchange() -> mischief::Result<()> {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
    .args([  "transform",
            "-v",
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
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
    Ok(())
}
