param (
    [ValidateSet($null,"-r")]
    $config = $null
)

Set-Location $PSScriptRoot
Set-Location ..
Set-Location crates/pyxis-py
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

pixi run cargo build  $config -p pyxis-py
pixi run maturin develop $config

Set-Location $PSScriptRoot
Set-Location ../../../
