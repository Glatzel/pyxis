Set-Location $PSScriptRoot
Set-Location ..
Set-Location vcpkg_deps

$triplet=Resolve-Path ./triplet
&./vcpkg/vcpkg.exe install --overlay-triplets=$triplet
