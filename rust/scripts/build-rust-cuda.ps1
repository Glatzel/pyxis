param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1
cargo build --profile $config -p pyxis-cuda
Set-Location $PSScriptRoot
Set-Location $ROOT
