Set-Location $PSScriptRoot
Set-Location ..
Set-Location vcpkg_deps

$triplet=Resolve-Path ./triplet
if($env:CI)
{
    vcpkg install --overlay-triplets=$triplet
}
else{
    &./vcpkg/vcpkg.exe install --overlay-triplets=$triplet
}
