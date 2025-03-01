param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug",
    [string]$version = "default"
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item dist/geotool*.whl -ErrorAction SilentlyContinue
Remove-Item crates/py-geotool/geotool/py_geotool.pyd -ErrorAction SilentlyContinue
Remove-Item crates/py-geotool/geotool/**__pycache__ -Recurse -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run -e $version cargo build --profile $config --lib
    Set-Location crates/py-geotool
    pixi run -e $version maturin build --out ../../dist --profile $config
}
else {
    pixi run -e $version cargo build --lib
    Set-Location crates/py-geotool
    pixi run -e $version maturin build --out ../../dist
}
