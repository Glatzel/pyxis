Set-Location $PSScriptRoot
Set-Location ..

if ($env:CI) {
    cargo +nightly bench
}
else {
    & $PSScriptRoot/set-env.ps1
    # pixi run cargo +nightly bench
    pixi run cargo bench
}
