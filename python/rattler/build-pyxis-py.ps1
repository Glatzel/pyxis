$ROOT = git rev-parse --show-toplevel
foreach ($whl in Get-ChildItem "$ROOT/dist/*.whl")
{
    pip install "$whl" -v
}
