Set-Location $PSScriptRoot
Set-Location ..
Set-Location vcpkg_deps

Remove-Item vcpkg -Recurse -Force -ErrorAction SilentlyContinue
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.bat
