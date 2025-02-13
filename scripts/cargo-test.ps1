Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

write-output "::group::nextest"
pixi run cargo +nightly llvm-cov --no-report --all-features --workspace nextest
$code = $LASTEXITCODE
Write-Output "::endgroup::"

write-output "::group::test"
pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --doc
$code = $code + $LASTEXITCODE
Write-Output "::endgroup::"

write-output "::group::report"
pixi run cargo +nightly llvm-cov report
Write-Output "::endgroup::"

write-output "::group::report lcov"
if ( $env:CI ) {
    pixi run cargo +nightly llvm-cov report --lcov --output-path lcov.info
}
Write-Output "::endgroup::"

$code = $code + $LASTEXITCODE
Write-Output $code
exit $code
