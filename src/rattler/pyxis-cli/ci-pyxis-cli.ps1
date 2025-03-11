Set-Location $PSScriptRoot
. ../scripts/utils.ps1
& "../../rust/scripts/build-cli.ps1" -config dist
Set-Location $PSScriptRoot
build_pkg
test_pkg