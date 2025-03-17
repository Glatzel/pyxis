$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

# only run cuda test in local machine
$python_root = Resolve-Path $PSScriptRoot/..
$env:PYTHONPATH = "$env:PYTHONPATH;$python_root"
if ($env:CI) {
    $markers = "not cuda"
}
else {
    $markers = ""
}

# run test
pixi run -e bench pytest `
    ./benches `
    -m $markers `
    --codspeed
Set-Location $ROOT
