param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
Set-Location crates/py-geotool
Remove-Item geotool/geotool.pyd -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config --lib
    pixi run maturin develop --profile $config
}
else {
    pixi run cargo build --lib
    pixi run maturin develop
}
