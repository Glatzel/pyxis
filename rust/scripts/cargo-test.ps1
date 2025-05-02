$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

& $PSScriptRoot/set-env.ps1
if ($env:CI) { $package = "-p", "pyxis", "-p", "pyxis-cli"}
else { $package = "-p", "pyxis", "-p", "pyxis-cli", "-p", "pyxis-cuda" }
Write-Output "::group::nextest"
cargo +nightly llvm-cov --no-report --all-features $package --branch nextest --no-fail-fast
$code = $LASTEXITCODE
Write-Output "::endgroup::"

Write-Output "::group::doctest"
cargo +nightly llvm-cov --no-report --all-features $package --branch --doc --no-fail-fast
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
    Write-Output "Test failed."
}
else {
    Write-Output "Test successed."
}
Write-Output "::endgroup::"
Set-Location $ROOT
exit $code
