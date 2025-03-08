param(
$package,
$filter)
Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

cargo bench -p $package -- $filter
Set-Location $PSScriptRoot
Set-Location ../../../
