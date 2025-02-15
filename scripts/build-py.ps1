param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug",
    [switch]$clean
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item dist/geotool*.whl -ErrorAction SilentlyContinue
Remove-Item crates/py-geotool/geotool/py_geotool.pyd -ErrorAction SilentlyContinue
Remove-Item crates/py-geotool/geotool/**__pycache__ -Recurse -ErrorAction SilentlyContinue

if ($clean) { cargo clean }
Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config --lib
    Set-Location crates/py-geotool
    pixi run maturin build --out ../../dist --profile $config
}
else {
    pixi run cargo build --lib
    Set-Location crates/py-geotool
    pixi run maturin build --out ../../dist
}
