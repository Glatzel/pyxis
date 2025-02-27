Set-Location $PSScriptRoot
Set-Location ..

Set-Location vcpkg_deps
$triplet=Resolve-Path ./triplet

Write-Output "::group::static"
&./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-windows
Write-Output "::endgroup::"

Write-Output "::group::dynamic"
&./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-windows-static
Write-Output "::endgroup::"