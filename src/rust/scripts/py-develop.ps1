param (
    [ValidateSet("develop","release")]
    $config = "develop"
)

Set-Location $PSScriptRoot
Set-Location ..
Set-Location crates/pyxis-py
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

pixi run cargo build --profile  $config -p pyxis-py
pixi run maturin develop --profile $config

Set-Location $PSScriptRoot
Set-Location ../../../
