param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot
. ../scripts/utils.ps1

& "$ROOT/python/scripts/build-cuda.ps1"
& "$ROOT/python/scripts/py-develop.ps1" -config $config
& "$ROOT/python/scripts/py-pytest.ps1"
& "$ROOT/python/scripts/build-python-whl.ps1" -config $config

Set-Location $PSScriptRoot
build_pkg
Set-Location $ROOT
