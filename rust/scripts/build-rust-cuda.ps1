param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$env:RUST_BACKTRACE=1
pixi run -e gpu cargo build --profile $config -p pyxis-cuda
Set-Location $PSScriptRoot
Set-Location $ROOT
