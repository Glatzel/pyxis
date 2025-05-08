$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo +stable clippy --fix --all-features -p pyxis -p pyxis-cli -p proj -- -D warnings
}
else {
    cargo clippy --fix --all -- -D warnings
}

Set-Location $ROOT
