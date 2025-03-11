param (
    [ValidateSet("debug","--profile dist")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
Set-Location crates/pyxis-py
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
pixi run cargo build  $config -p pyxis-py
pixi run maturin develop $config

Set-Location $PSScriptRoot
Set-Location ../../../
