$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
if ($env:CI) {
    pixi install --no-progress
}
else {
    pixi install
}

if ($IsWindows) {
    if (-not $env:CI) {
        $path = pixi run -e default vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($path) {
            $path = join-path $path 'Common7\Tools\vsdevcmd.bat'
            if (test-path $path) {
                cmd /s /c """$path"" $args && set" | where { $_ -match '(\w+)=(.*)' } | foreach {
                    $null = new-item -force -path "Env:\$($Matches[1])" -value $Matches[2]
                }
            }
        }
    }

    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $nvcc_path = Resolve-Path $PSScriptRoot/../.pixi/envs/gpu/Library/bin
    $env:CUDA_ROOT = Resolve-Path $PSScriptRoot/../.pixi/envs/gpu/Library
    $env:PATH = "$nvcc_path;$pkg_config_exe;$env:PATH"
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
    $env:CUDA_ROOT = Resolve-Path $PSScriptRoot/../.pixi/envs/gpu
    $env:Path = "$pkg_config_exe" + ":" + "$env:Path"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
    Copy-Item ./.pixi/envs/default/proj/x64-linux-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}

Set-Location $current_dir
