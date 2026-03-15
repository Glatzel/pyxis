Set-Location $PSScriptRoot/..
git submodule update --init --recursive
pixi install
if ($IsWindows) {
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin);$env:Path"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library)"
}
if ($IsMacOS) {
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:Path"
    $env:DYLD_LIBRARY_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:DYLD_LIBRARY_PATH"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default)"
}
if ($IsLinux) {
    sudo apt update
    sudo apt install -y libudev-dev libc6-dev
}
if ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    if (env:CI) {
        sudo apt-get update
        sudo apt install -y libudev-dev libc6-dev
    }
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:Path"
    $env:LD_LIBRARY_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:LD_LIBRARY_PATH"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default)"
}
if ($IsLinux -and ($(uname -m) -eq 'aarch64' )) {
    if (env:CI) {
        sudo apt-get update
        sudo apt install -y libudev-dev libc6-dev
    }
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:Path"
    $env:LD_LIBRARY_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:LD_LIBRARY_PATH"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default)"
}
