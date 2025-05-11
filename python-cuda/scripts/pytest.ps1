$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

# only run cuda test in local machine
$python_root = Resolve-Path $PSScriptRoot/../src
$pyxis = Resolve-Path $ROOT/python
$env:PYTHONPATH = "$python_root;$pyxis;$env:PYTHONPATH"

# run test
pixi run -e local-test pytest `
    ./tests `
    -v `
    --durations=10 `
    --junitxml=junit.xml `
    -o junit_family=legacy `
    --cov `
    --cov-report term `
    --cov-report=xml:coverage.xml `
    --cov-report=html:htmlcov
Set-Location $ROOT
