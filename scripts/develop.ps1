param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug",
    [switch]$clean
)

Set-Location $PSScriptRoot
Set-Location ..
Set-Location crates/py-geotool
Remove-Item geotool/geotool.pyd -ErrorAction SilentlyContinue

if ($clean) { cargo clean }
Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run -e ci-py cargo build --profile $config --lib
    pixi run -e ci-py maturin develop --profile $config
}
else {
    pixi run cargo build --lib
    pixi run maturin develop
}
