param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug",
    [string]$version = "default"
)

Set-Location $PSScriptRoot
Set-Location ..
Set-Location crates/py-geotool
Remove-Item geotool/geotool.pyd -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run -e $version cargo build --profile $config --lib
    pixi run -e $version maturin develop --profile $config
}
else {
    pixi run -e $version cargo build --lib
    pixi run -e $version maturin develop
}
