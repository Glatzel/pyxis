Set-Location $PSScriptRoot
Set-Location ..

# zygs
& ./dist/geotool/geotool.exe -v `
    transform -n zygs -x 469704.6693 -y 2821940.796 -z 0 -o plain `
    datum-compense --hb 400 -r 6378137 --x0 500000 --y0 0 `
    proj --from "+proj=tmerc +lat_0=0 +lon_0=118.5 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs" `
    --to "+proj=longlat +datum=WGS84 +no_defs +type=crs"
Write-Output ""

# jxws
& ./dist/geotool/geotool.exe `
    transform -n jxws -x 121.091701 -y 30.610765 -z 0 -o json `
    crypto --from "wgs84" --to "gcj02"

# 
& ./dist/geotool/geotool.exe `
transform -x 116.3913318 -y 39.9055625 -z 0 `
crypto --from "wgs84" --to "gcj02"

Set-Location $PSScriptRoot
Set-Location ..
