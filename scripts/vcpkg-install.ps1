Set-Location $PSScriptRoot
Set-Location ..

Set-Location vcpkg_deps
$triplet=Resolve-Path ./triplet
if($IsWindows){
    Write-Output "::group::dynamic"
    &./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-windows --x-install-root ./vcpkg_installed/dynamic
    Write-Output "::endgroup::"
    
    Write-Output "::group::static"
    &./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-windows-static --x-install-root ./vcpkg_installed/static
    Write-Output "::endgroup::"
    
}elseif ($IsLinux) {
    Write-Output "::group::dynamic"
    &./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-linux --x-install-root ./vcpkg_installed/dynamic
    Write-Output "::endgroup::"
    
    Write-Output "::group::static"
    &./vcpkg/vcpkg.exe install --overlay-triplets=$triplet --triplet x64-linux-static --x-install-root ./vcpkg_installed/static
    Write-Output "::endgroup::"
}
