Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo clippy --all-targets --all-features
}
else {
    pixi run cargo clippy --fix --all-targets --all-features
}
