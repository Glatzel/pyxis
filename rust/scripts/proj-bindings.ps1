$ROOT = git rev-parse --show-toplevel
&$PSScriptRoot/setup.ps1
Set-Location $PSScriptRoot/..
$env:RUSTFLAGS = "-Dwarnings"
$env:UPDATE = "true"
cargo build -p proj-sys
$env:UPDATE = "false"
Set-Location $ROOT
