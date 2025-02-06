Set-Location $PSScriptRoot
Set-Location ..

& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo clippy --all-targets --all-features
}
else {
    cargo fmt
    cargo clippy --fix --all-targets
}