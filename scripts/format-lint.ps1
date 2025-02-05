Set-Location $PSScriptRoot
Set-Location ..

& $PSScriptRoot/set-env.ps1
cargo fmt
cargo clippy --fix --all-targets
