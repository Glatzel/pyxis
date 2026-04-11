$ROOT = git rev-parse --show-toplevel
& $PSScriptRoot/setup.ps1
Set-Location $PSScriptRoot/..
$env:RUSTFLAGS = "-Dwarnings"
cargo build -p proj-sys --features bindgen
Set-Location $ROOT
