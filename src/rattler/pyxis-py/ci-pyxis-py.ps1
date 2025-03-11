param (
    [ValidateSet("develop","release")]
    $config = "develop"
)

Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "../../rust/scripts/py-develop.ps1" -config $config
& "../../rust/scripts/py-pytest.ps1"
& "../../rust/scripts/build-py.ps1" -config $config

Set-Location $PSScriptRoot
build_pkg
test_pkg