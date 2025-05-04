if ($IsLinux) {
    $ROOT = git rev-parse --show-toplevel
    Set-Location $PSScriptRoot/..
    $env:RUSTFLAGS = "-Dwarnings"
    & $PSScriptRoot/set-env.ps1
    cargo build  --all-features -p proj-sys
    Set-Location $ROOT
    
}
