param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "$ROOT/scripts/py-develop.ps1" -config $config
& "$ROOT/scripts/py-pytest.ps1"
& "$ROOT/scripts/build-py.ps1" -config $config

Set-Location $PSScriptRoot
build_pkg
Set-Location $ROOT
