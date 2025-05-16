param($install = "./dist")
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$config = "-DCMAKE_BUILD_TYPE=Release"

# create install dir
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = Resolve-Path $install
$install = "$install".Replace('\', '/')
$install = "-DCMAKE_INSTALL_PREFIX=$install"

# create ptx output dir
Remove-Item ./build -Recurse -Force -ErrorAction SilentlyContinue
New-Item ./build/ptx -ItemType Directory -ErrorAction SilentlyContinue

# build
if ($IsWindows) { pixi run cmake -G "Visual Studio 17 2022" -B build $install $config -DBUILD_CUDA=ON }
else { pixi run cmake -B build $install $config -DBUILD_CUDA=ON }
pixi run cmake --build build --target install

Set-Location $ROOT
