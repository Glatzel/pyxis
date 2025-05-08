$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
pixi install --no-progress

if ($IsWindows) {
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $env:PATH = "$pkg_config_exe;$env:PATH"
    $env:PKG_CONFIG_PATH = Resolve-Path "./.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig"
    Copy-Item ./.pixi/envs/default/proj/x64-windows-static/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsMacOS) {
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/bin
    $env:Path = "$pkg_config_exe" + ":" + "$env:Path"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/lib/pkgconfig
    Copy-Item ./.pixi/envs/default/proj/arm64-osx-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/bin
    $env:Path = "$pkg_config_exe" + ":" + "$env:Path"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
    Copy-Item ./.pixi/envs/default/proj/x64-linux-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}

Set-Location $current_dir
