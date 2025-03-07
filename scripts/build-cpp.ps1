param(
    $config,
    $install="./dist/cpp"
)

if ($config) {
    $config = "-DCMAKE_BUILD_TYPE=Release"
}
if ($install) {
    if (-not (Test-Path $install)) {
        New-Item $install -ItemType Directory
    }
    $install = Resolve-Path $install
    $install = "$install".Replace('\', '/')
    $install = "-DCMAKE_INSTALL_PREFIX=$install"
}
Set-Location $PSScriptRoot
Set-Location ..
Set-Location ./src/cpp

cmake . -B build $install $config
if ($install) {
    cmake --build build --target install
}
else {
    cmake --build build
}
Set-Location $PSScriptRoot
Set-Location ..