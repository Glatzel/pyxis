New-Item $env:PREFIX/bin/pyxis-cli -ItemType Directory
Copy-Item "$env:RECIPE_DIR/../../../dist/cli/static/pyxis.exe" "$env:PREFIX/bin/pyxis-cli/pyxis.exe"
