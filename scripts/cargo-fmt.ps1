$ROOT = git rev-parse --show-toplevel
Set-Location $ROOT
& $PSScriptRoot/set-env.ps1
if ($env:CI) {
    cargo +nightly fmt --all -- --check
}
else {
    pixi run cargo +nightly fmt --all
}

Set-Location $ROOT
