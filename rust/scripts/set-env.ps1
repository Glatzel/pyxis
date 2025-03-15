param (
    [ValidateSet("static", "dynamic")]
    [string]$link = "static"
)
$ROOT = git rev-parse --show-toplevel
$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    # set libproj find path for pkgconfig
    if ($link -eq 'static') {
        $env:PKG_CONFIG_PATH = "$ROOT/vcpkg/installed/static/x64-windows-static/lib/pkgconfig"
    }
    else {
        $env:PKG_CONFIG_PATH = "$ROOT/vcpkg/installed/dynamic/x64-windows/lib/pkgconfig"
    }
    # copy proj.db to cli src
    Copy-Item $ROOT/rust/.pixi/envs/default/Library/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    # copy proj.db to cli src
    pixi install
    Copy-Item $ROOT/rust/.pixi/envs/default/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
Set-Location $current_dir
