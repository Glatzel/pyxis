function build_pkg {
    Write-Output "::group::build"
    pixi run rattler-build build
    Write-Output "::endgroup::"
}
