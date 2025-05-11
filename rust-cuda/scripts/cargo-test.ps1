$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

& $PSScriptRoot/setup.ps1
pixi run cargo +nightly llvm-cov nextest --no-report --all --branch --no-fail-fast
$code = $LASTEXITCODE
pixi run cargo +nightly llvm-cov --no-report --all $package --branch --no-fail-fast --doc
$code = $code + $LASTEXITCODE
cargo +nightly llvm-cov report
cargo +nightly llvm-cov report --lcov --output-path lcov.info
$code = $code + $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "Test failed." -ForegroundColor Red
}
else {
    Write-Host "Test succeeded." -ForegroundColor Green
}
Set-Location $ROOT
exit $code
