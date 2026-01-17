Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $bin = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $env:Path = "$bin" + ";" + "$env:Path"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/lib)"
}
if ($IsMacOS) {
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:Path"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)"
}
if ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:Path"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)"
}
if ($IsLinux -and ($(uname -m) -eq 'aarch64' )) {
    $env:Path = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)`:$env:Path"
    $env:PROJ_ROOT = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib)"
}
