$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

& $PSScriptRoot/set-env.ps1
git submodule update --init --recursive
if ($env:CI) { $package = "-p", "pyxis", "-p", "pyxis-cli", "-p", "proj" }
else { $package = "-p", "pyxis", "-p", "pyxis-cli", "-p", "pyxis-cuda", "-p", "proj" }

if ($IsWindows) {
    $env:PROJ_DATA = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/share/proj
}
if ($IsLinux) {
    $env:PROJ_DATA = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/share/proj
}
if ($IsMacOS) {
    $env:PROJ_DATA = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/share/proj
}
Write-Output "::group::nextest"
pixi run cargo +nightly llvm-cov nextest --no-report --all-features $package --branch --no-fail-fast
$code = $LASTEXITCODE
Write-Output "::endgroup::"

Write-Output "::group::doctest"
pixi run cargo +nightly llvm-cov --no-report --all-features $package --branch --no-fail-fast --doc
$code = $code + $LASTEXITCODE
Write-Output "::endgroup::"

Write-Output "::group::report"
cargo +nightly llvm-cov report
Write-Output "::endgroup::"

Write-Output "::group::lcov"
if ( $env:CI ) {
    cargo +nightly llvm-cov report --lcov --output-path lcov.info
}
Write-Output "::endgroup::"

Write-Output "::group::result"
$code = $code + $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "Test failed." -ForegroundColor Red
}
else {
    Write-Host "Test succeeded." -ForegroundColor Green
}
Write-Output "::endgroup::"
Set-Location $ROOT
exit $code
