Set-Location $PSScriptRoot
Set-Location ..

if ($env:CI) {
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features
}
else {
    & $PSScriptRoot/set-env.ps1
    pixi run cargo fmt -- --order-imports
    pixi run cargo clippy --fix --all-targets --all-features 
}
