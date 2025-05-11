$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$config = "-DCMAKE_BUILD_TYPE=Release"

# create install dir
$install = "./dist"
Remove-Item $install -Recurse -ErrorAction SilentlyContinue
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = Resolve-Path $install
$install = "$install".Replace('\', '/')
$install = "-DCMAKE_INSTALL_PREFIX=$install"

# create ptx output dir
New-Item ./build/ptx -ItemType Directory -ErrorAction SilentlyContinue

# find visual studio
if (-not $env:CI) {
    $path = pixi run vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($path) {
        $path = join-path $path 'Common7\Tools\vsdevcmd.bat'
        if (test-path $path) {
            cmd /s /c """$path"" $args && set" | where { $_ -match '(\w+)=(.*)' } | foreach {
                $null = new-item -force -path "Env:\$($Matches[1])" -value $Matches[2]
            }
        }
    }
}

# build
pixi run cmake -B build $install $config -DBUILD_CUDA=ON
pixi run cmake --build build --target install

# pack output files
if ($IsWindows) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cuda-windows-x64.7z" "./dist/"
}if ($IsLinux) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cuda-linux-x64.7z" "./dist/"
}
Set-Location $ROOT
