param($filter)
Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

pixi run cargo bench --package pyxis-algorithm -- $filter
