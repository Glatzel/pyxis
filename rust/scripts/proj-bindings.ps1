$ROOT = git rev-parse --show-toplevel
& $PSScriptRoot/setup.ps1
Set-Location $PSScriptRoot/..
$env:UPDATE_PROJ_BINDGEN = "1"
$env:RUSTFLAGS = "-Dwarnings"
cargo build -p proj-sys
Set-Location $ROOT
