$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo +stable clippy --all-features --all-targets
}
else {
    pixi run cargo clippy --fix --all-targets --all-features
}

Set-Location $ROOT
