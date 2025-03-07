param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config -p pyxis-py
    pixi run maturin develop --profile $config
}
else {
    pixi run cargo build -p pyxis-py
    pixi run maturin develop
}
Set-Location $PSScriptRoot
Set-Location ../../../
