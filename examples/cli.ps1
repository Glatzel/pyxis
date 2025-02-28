Set-Location $PSScriptRoot
Set-Location ..

if ($IsWindows) {
    $geotool = "./geotool.exe"
    $links = "static", "dynamic"
}
elseif ($IsLinux) {
    $geotool = "./geotool" 
    $links = "static" 
}

foreach ($link in $links) {
    Set-Location "dist/cli/$link"
    Write-Output "::group::Run $link cli"

    # Zhengyong expressway Dehua east interchange
    & "$geotool" -v `
        transform -n "Zhengyong expressway Dehua east interchange" -x 469704.6693 -y 2821940.796 -z 0 -o plain `
        datum-compense --hb 400 -r 6378137 --x0 500000 --y0 0 `
        proj --from "+proj=tmerc +lat_0=0 +lon_0=118.5 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs" `
        --to "+proj=longlat +datum=WGS84 +no_defs +type=crs"
    Write-Output ""

    # Jiaxing bump station
    & "$geotool" `
        transform -n "Jiaxing bump station" -x 121.091701 -y 30.610765 -z 0 -o json `
        crypto --from "wgs84" --to "gcj02"
    Write-Output ""

    # Tian'anmen national flag
    & "$geotool" `
        transform -n "Tian'anmen national flag" -x 116.3913318 -y 39.9055625 -z 0 -o plain `
        crypto --from "wgs84" --to "gcj02"
    Write-Output ""

    # proj crate
    & "$geotool" `
        transform -n "proj crate" -x 4760096.421921 -y 3744293.729449 -z 0 -o plain `
        proj --from "EPSG:2230" --to "EPSG:26946"

    Set-Location $PSScriptRoot
    Set-Location ..
    Write-Output "::endgroup::"
}