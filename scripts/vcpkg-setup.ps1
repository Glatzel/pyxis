Set-Location $PSScriptRoot
Set-Location ..
Set-Location vcpkg_deps

Remove-Item vcpkg -Recurse -Force -ErrorAction SilentlyContinue
git clone https://github.com/microsoft/vcpkg.git

if ($IsWindows) {
    ./vcpkg/bootstrap-vcpkg.bat
    
}
elseif ($IsLinux) {
    ./vcpkg/bootstrap-vcpkg.sh
}
