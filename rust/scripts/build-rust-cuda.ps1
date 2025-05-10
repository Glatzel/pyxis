param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

pixi run -e gpu cargo build --profile $config -p pyxis-cuda
Set-Location $PSScriptRoot
Set-Location $ROOT
