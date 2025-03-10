Set-Location $PSScriptRoot
Set-Location ..

Set-Location vcpkg_deps

# use custom triplet
$triplet = Resolve-Path ./triplet

# install static dependency
Write-Output "::group::dynamic"
&./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-windows --x-install-root ./vcpkg_installed/dynamic
Write-Output "::endgroup::"

# install static dependency
Write-Output "::group::static"
&./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-windows-static --x-install-root ./vcpkg_installed/static
Write-Output "::endgroup::"
