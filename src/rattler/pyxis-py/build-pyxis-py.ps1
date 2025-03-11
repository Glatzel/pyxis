New-Item $env:PREFIX/bin/pyxis-cli -ItemType Directory
foreach ($whl in Get-ChildItem "$env:RECIPE_DIR/../../../dist/*.whl")
{
    pip install "$whl" -v
}
