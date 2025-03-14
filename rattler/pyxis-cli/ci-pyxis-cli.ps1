param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "$ROOT/rust/scripts/build-rust-cli.ps1" -config $config
& "$ROOT/rust/pyxis-cli/examples/cli.ps1"

Set-Location $PSScriptRoot
build_pkg
New-Item $ROOT/dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $ROOT/dist
Set-Location $ROOT
