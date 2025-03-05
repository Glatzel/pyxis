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
    pixi run cargo build --profile $config -p pyxis-cuda
}
else {
    pixi run cargo build -p pyxis-cuda
}
