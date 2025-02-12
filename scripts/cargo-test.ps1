Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1


if ( $env:CI ) {
    pixi run -e ci-cli cargo +nightly llvm-cov --no-report --all-features --workspace nextest
    $code = $LASTEXITCODE
    pixi run -e ci-cli cargo +nightly llvm-cov --no-report --all-features --workspace --doc
    $code = $code + $LASTEXITCODE
    pixi run -e ci-cli cargo +nightly llvm-cov report
    pixi run -e ci-cli cargo +nightly llvm-cov report --lcov --output-path lcov.info
}
else {
    pixi run cargo +nightly llvm-cov --no-report --all-features --workspace nextest
    $code = $LASTEXITCODE
    pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --doc
    $code = $code + $LASTEXITCODE
    pixi run cargo +nightly llvm-cov report
}
$code = $code + $LASTEXITCODE
Write-Output $code
exit $code
