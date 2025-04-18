$ROOT = git rev-parse --show-toplevel
$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $env:PKG_CONFIG_PATH = Resolve-Path "./.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig"
    Copy-Item ./.pixi/envs/default/proj/x64-windows-static/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    Copy-Item ./.pixi/envs/default/proj/x64-linux-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
Set-Location $current_dir
