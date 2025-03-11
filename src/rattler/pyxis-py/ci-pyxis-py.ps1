param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "../../rust/scripts/py-develop.ps1" -config dist
& "../../rust/scripts/py-pytest.ps1" -config dist
& "../../rust/scripts/build-py.ps1" -config dist

Set-Location $PSScriptRoot
build_pkg
test_pkg