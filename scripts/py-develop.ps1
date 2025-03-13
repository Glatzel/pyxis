param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
Set-Location src/python
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

pixi run cargo build --profile  $config -p pyxis-py
pixi run maturin develop --profile $config

Set-Location $ROOT
