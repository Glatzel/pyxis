$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$env:RUSTFLAGS = "-Dwarnings"
& $PSScriptRoot/set-env.ps1
$env:OUT_DIR=resolve-path "$PSScriptRoot/../crates/proj-sys/src"
cargo build  --all-features -p proj-sys
Set-Location $ROOT
