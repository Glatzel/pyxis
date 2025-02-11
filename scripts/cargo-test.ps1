Set-Location $PSScriptRoot
Set-Location ..
$ErrorActionPreference = "Stop"
$pkg_config = Resolve-Path .pixi/envs/dev/Library/bin
$dll_path = Resolve-Path vcpkg_deps/vcpkg_installed/x64-windows/bin
$env:PATH = $env:PATH + ";$pkg_config;$dll_path"
$env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/x64-windows/lib/pkgconfig

pixi run cargo +nightly llvm-cov --no-report --all-features --workspace nextest
$code=$LASTEXITCODE
pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --doc
$code=$code -and $LASTEXITCODE
if ( $env:CI ) {
    pixi run cargo +nightly llvm-cov report
    pixi run cargo +nightly llvm-cov report --lcov --output-path lcov.info
}
else {
    pixi run cargo +nightly llvm-cov report
}
$code=$code -and $LASTEXITCODE
if (-not $code){
    exit 1
}
