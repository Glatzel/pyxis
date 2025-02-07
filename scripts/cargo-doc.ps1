Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

cargo doc --no-deps --all