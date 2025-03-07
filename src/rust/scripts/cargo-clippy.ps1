Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo clippy --all-features -p pyxis -p pyxis-cli -p pyxis-py
}
else {
    pixi run cargo clippy --fix --all-targets --all-features
}
Set-Location $PSScriptRoot
Set-Location ../../../
