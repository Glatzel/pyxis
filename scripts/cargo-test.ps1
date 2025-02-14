Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

pixi run cargo +nightly llvm-cov --no-report --all-features --workspace nextest
$code = $LASTEXITCODE
pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --doc
$code = $code + $LASTEXITCODE
pixi run cargo +nightly llvm-cov report

if ( $env:CI ) {
    pixi run cargo +nightly llvm-cov report --lcov --output-path lcov.info
}

$code = $code + $LASTEXITCODE
Write-Output $code
exit $code
