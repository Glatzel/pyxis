param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..

# add nvcc to path
if($IsWindows){
    $env:PATH="$env:PATH;C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
}

if ($config -ne "debug") {
    cargo build --profile $config -p pyxis-cuda
}
else {
    cargo build -p pyxis-cuda
}
