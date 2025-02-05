Set-Location $PSScriptRoot
Set-Location ..
Set-Location vcpkg_deps
git clone https://github.com/microsoft/vcpkg.git --depth 1
./vcpkg/bootstrap-vcpkg.bat
