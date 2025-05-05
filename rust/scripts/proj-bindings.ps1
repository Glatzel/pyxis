$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$env:RUSTFLAGS = "-Dwarnings"
$env:UPDATE = "true"
& $PSScriptRoot/set-env.ps1
cargo build  --all-features -p proj-sys
$env:UPDATE = "false"
Set-Location $ROOT
