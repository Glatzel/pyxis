param (
    [ValidateSet("develop","release")]
    $config = "develop"
)

Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "$PSScriptRoot/../../rust/scripts/build-cli.ps1" -config $config
& "$PSScriptRoot/../../rust/crates/pyxis-cli/examples/cli.ps1"

Set-Location $PSScriptRoot
build_pkg
New-Item $PSScriptRoot/../../../dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $PSScriptRoot/../../../dist
