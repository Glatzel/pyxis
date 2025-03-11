function build_pkg {
    Write-Output "::group::build"
    pixi run rattler-build build
    Write-Output "::endgroup::"
}
function test_pkg {
    Write-Output "::group::test"
    if ($IsWindows) {
        foreach ($conda_file in Get-ChildItem "./output/win-64/*.conda") {
            pixi run rattler-build test --package-file $conda_file
        }
    }if ($IsLinux) {
        foreach ($conda_file in Get-ChildItem "./output/linux-64/*.conda") {
            pixi run rattler-build test --package-file $conda_file
        }
    }
    Write-Output "::endgroup::"
}
