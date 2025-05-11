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
if ($env:CI) {
    $vsPath = & pixi run vswhere `
        -latest `
        -requires Microsoft.Component.MSBuild `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    & "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
}

# build
pixi run cmake -G "Visual Studio 17 2022" -B build $install $config -DBUILD_CUDA=ON
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
