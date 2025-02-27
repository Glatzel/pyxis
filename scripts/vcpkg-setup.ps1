Set-Location $PSScriptRoot
Set-Location ..
Set-Location vcpkg_deps
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.bat
