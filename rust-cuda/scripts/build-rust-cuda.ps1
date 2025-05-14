param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
&$PSScriptRoot/setup.ps1
cargo build --profile $config -p pyxis-cuda
Set-Location $PSScriptRoot
Set-Location $ROOT
