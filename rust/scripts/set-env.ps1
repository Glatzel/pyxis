$ROOT = git rev-parse --show-toplevel
$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    # set libproj find path for pkgconfig
    $env:PKG_CONFIG_PATH = Resolve-Path "./vcpkg/installed/x64-windows-static/lib/pkgconfig"

    # copy proj.db to cli src
    Copy-Item ./.pixi/envs/default/Library/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    # copy proj.db to cli src
    pixi install
    Copy-Item ./.pixi/envs/default/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
Set-Location $current_dir
