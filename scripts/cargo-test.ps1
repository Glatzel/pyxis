Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1
New-Item ./target/llvm-cov-target/debug -ItemType Directory -ErrorAction SilentlyContinue
$os= Get-ComputerInfo -Property OsName
switch -Wildcard ($os) {
    "*Windows*" {
        Copy-Item "./vcpkg_deps/vcpkg_installed/static/x64-windows-static/share/proj/proj.db" ./target/llvm-cov-target/debug
        Write-Output "::group::nextest"
        pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --branch nextest
        $code = $LASTEXITCODE
        Write-Output "::endgroup::"

        Write-Output "::group::doctest"
        pixi run cargo +nightly llvm-cov --no-report --all-features --workspace --branch --doc
        $code = $code + $LASTEXITCODE
        Write-Output "::endgroup::"

        Write-Output "::group::report"
        pixi run cargo +nightly llvm-cov report
        Write-Output "::endgroup::"

        Write-Output "::group::lcov"
        if ( $env:CI ) {
            pixi run cargo +nightly llvm-cov report --lcov --output-path lcov.info
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
        exit $code
    }
    "*Linux*" {
        Write-Output "::group::nextest"
        cargo +nightly llvm-cov --no-report --all-features --workspace --branch nextest
        $code = $LASTEXITCODE
        Write-Output "::endgroup::"

        Write-Output "::group::doctest"
        cargo +nightly llvm-cov --no-report --all-features --workspace --branch --doc
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
        exit $code
    }
    default {
        Write-Error "Unsupported system $os"
        exit 1
    }
}