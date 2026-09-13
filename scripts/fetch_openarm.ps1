$ErrorActionPreference = 'Stop'
$destination = Join-Path (Split-Path $PSScriptRoot -Parent) 'third_party/openarm_mujoco'
if (Test-Path -LiteralPath $destination) { throw 'Destination already exists; inspect it instead of overwriting.' }
git clone https://github.com/enactic/openarm_mujoco.git $destination
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
git -C $destination checkout --detach 161039cd74ea8675fb8197836fe5674659825c75
if ($LASTEXITCODE -ne 0) { throw 'Pinning failed' }
