$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

$python_root = Resolve-Path $PSScriptRoot/..
$env:PYTHONPATH = "$env:PYTHONPATH;$python_root"

# run test
pixi run pytest `
    ./tests `
    -v `
    --durations=10 `
    --junitxml=junit.xml `
    -o junit_family=legacy `
    --cov `
    --cov-report term `
    --cov-report=xml:coverage.xml `
    --color=yes
Set-Location $ROOT
