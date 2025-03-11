param (
    [ValidateSet("develop","release")]
    $config = "develop"
)

Set-Location $PSScriptRoot
Set-Location ..

# add nvcc to path
if($IsWindows){
    $env:PATH="$env:PATH;C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
}
cargo build --profile $config -p pyxis-cuda
Set-Location $PSScriptRoot
Set-Location ../../../
