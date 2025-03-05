param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..

# add nvcc to path
if($IsWindows){
    $env:PATH="$env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
}

if ($config -ne "debug") {
    pixi run cargo build --profile $config -p pyxis-cuda
}
else {
    pixi run cargo build -p pyxis-cuda
}
