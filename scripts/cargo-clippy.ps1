Set-Location $PSScriptRoot
Set-Location ..

if ($env:CI) {
    cargo clippy --all-targets --all-features
}
else {
    & $PSScriptRoot/set-env.ps1
    pixi run cargo clippy --fix --all-targets --all-features 
}
