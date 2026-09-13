$ErrorActionPreference = 'Stop'
$destination = Join-Path (Split-Path $PSScriptRoot -Parent) 'third_party/openpi'
if (Test-Path -LiteralPath $destination) { throw 'Destination exists; inspect it rather than overwrite.' }
git clone --no-checkout https://github.com/Physical-Intelligence/openpi.git $destination
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
git -C $destination checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
if ($LASTEXITCODE -ne 0) { throw 'Pinning failed' }
# Source only. No dependency install, submodules, checkpoint downloads, or training.
