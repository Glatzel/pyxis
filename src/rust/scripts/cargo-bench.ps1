param($filter)
Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

cargo bench -p pyxis -p pyxis-cuda -- $filter
Set-Location $PSScriptRoot
Set-Location ../../../
