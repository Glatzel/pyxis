$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
if ($env:CI) {
    pixi install --no-progress
}
else {
    pixi install
}

if ($IsWindows) {
    Copy-Item ./.pixi/envs/default/proj/x64-windows-static/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsMacOS) {
    Copy-Item ./.pixi/envs/default/proj/arm64-osx-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    Copy-Item ./.pixi/envs/default/proj/x64-linux-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}

Set-Location $current_dir
