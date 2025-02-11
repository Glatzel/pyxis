Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --branch nextest
$code = $LASTEXITCODE
pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --branch --doc
$code = $code + $LASTEXITCODE
if ( $env:CI ) {
    pixi run cargo +nightly llvm-cov report
    pixi run cargo +nightly llvm-cov report --lcov --output-path lcov.info
}
else { pixi run cargo +nightly llvm-cov report }
$code = $code + $LASTEXITCODE
Write-Output $code
exit $code
