$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$env:RUSTFLAGS = "-Dwarnings"
$env:UPDATE = "true"
& $PSScriptRoot/setup.ps1
cargo build -p proj-sys
$env:UPDATE = "false"
Set-Location $ROOT
