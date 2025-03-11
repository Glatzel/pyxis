param (
    [ValidateSet("","-r")]
    [string]$config = ""
)

Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "../../rust/scripts/py-develop.ps1" $config
& "../../rust/scripts/py-pytest.ps1"
& "../../rust/scripts/build-py.ps1" $config

Set-Location $PSScriptRoot
build_pkg
test_pkg