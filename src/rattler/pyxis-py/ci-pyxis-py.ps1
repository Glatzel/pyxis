Set-Location $PSScriptRoot
. ../scripts/utils.ps1
& "../../rust/scripts/build-py.ps1" -config dist
Set-Location $PSScriptRoot
build_pkg
test_pkg