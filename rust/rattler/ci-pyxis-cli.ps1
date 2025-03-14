param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot

& "$ROOT/rust/scripts/build-rust-cli.ps1" -config $config
& "$ROOT/rust/crates/pyxis-cli/examples/cli.ps1"

Set-Location $PSScriptRoot
pixi run rattler-build build
New-Item $ROOT/dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $ROOT/dist
Set-Location $ROOT
