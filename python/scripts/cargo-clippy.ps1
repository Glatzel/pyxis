$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
# pixi run cargo +stable clippy --fix --all
# pixi run cargo +stable clippy --all
Set-Location $ROOT
