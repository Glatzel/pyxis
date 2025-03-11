Set-Location $PSScriptRoot

& "../../rust/scripts/build-cli.ps1" -config dist

Write-Output "::group::build"
pixi run rattler-build build
Write-Output "::endgroup::"

Write-Output "::group::test"
    if ($IsWindows) {
        foreach ($conda_file in Get-ChildItem "./output/win-64/*.conda") {
            pixi run rattler-build test --package-file $conda_file
        }
    }
Write-Output "::endgroup::"