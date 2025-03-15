param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot

& "$PSScriptRoot/../scripts/build-rust-cli.ps1" -config $config
& "$PSScriptRoot/../crates/pyxis-cli/examples/cli.ps1"

Set-Location $PSScriptRoot
pixi run rattler-build build
New-Item $PSScriptRoot/../dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $PSScriptRoot/../dist
Set-Location $ROOT
