param (
    [ValidateSet($null,"-r")]
    $config = $null
)

Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "$PSScriptRoot/../../rust/scripts/build-cli.ps1" -config dist
& "$PSScriptRoot/../../rust/crates/pyxis-cli/examples/cli.ps1"

Set-Location $PSScriptRoot
build_pkg
test_pkg
New-Item $PSScriptRoot/../../../dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $PSScriptRoot/../../../dist