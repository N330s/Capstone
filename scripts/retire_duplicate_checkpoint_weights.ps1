# One-time, hash-verified cleanup of four redundant weight files only.
# Default is read-only. No recursive deletion, optimizer deletion or hard links.
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outputRoot = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot 'outputs')).Path

function Resolve-WorkspaceFile([string]$relative) {
    $resolved = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot $relative)).Path
    if (-not $resolved.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'File escaped workspace'
    }
    $entry = Get-Item -LiteralPath $resolved -Force
    if ($entry.PSIsContainer) { throw 'Exact regular file required' }
    $cursor = $entry
    while ($null -ne $cursor) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point refused' }
        if ($cursor.FullName -eq $workspaceRoot) { break }
        $cursor = Get-Item -LiteralPath (Split-Path -Parent $cursor.FullName) -Force
    }
    return $resolved
}

function Write-Receipt([string]$path, $value) {
    # Generated audit data, not source-file editing.
    $payload = ConvertTo-Json -InputObject $value -Depth 12
    [IO.File]::WriteAllText($path, $payload, (New-Object Text.UTF8Encoding($false)))
}

$pairs = @(
    @{
        Remove = 'outputs\recovery_pipeline_20261019\pilot_v1\training\checkpoints\step_0000016_best\pretrained_model\model.safetensors'
        Keep = 'outputs\recovery_pipeline_20261019\pilot_v1\training\checkpoints\step_0000016_final\pretrained_model\model.safetensors'
        Hash = '4ca09cb5057d1e8497b772074ec034f587cb37bf8b881582144ae10bcae783aa'
    },
    @{
        Remove = 'outputs\recovery_sampling_20261019\correction_400\training\checkpoints\step_0000400_periodic\pretrained_model\model.safetensors'
        Keep = 'outputs\recovery_sampling_20261019\correction_400\training\checkpoints\step_0000400_final\pretrained_model\model.safetensors'
        Hash = 'ce08bf54b85d9666cb0428a29b933739751879e63da75286b46a5b7967da0189'
    },
    @{
        Remove = 'outputs\recovery_sampling_20261019\uniform_400\training\checkpoints\step_0000400_best\pretrained_model\model.safetensors'
        Keep = 'outputs\recovery_sampling_20261019\uniform_400\training\checkpoints\step_0000400_final\pretrained_model\model.safetensors'
        Hash = '93cbe1dffe7416a57f8eee6d8f8da8ce8b9f99f78c49228bb56acd580126fc97'
    },
    @{
        Remove = 'outputs\v2c30_sft_until_0800_v1\checkpoints\step_0011830_recovery_initial\pretrained_model\model.safetensors'
        Keep = 'outputs\v2c30_sft_overnight_v1\checkpoints\step_0011830_final\pretrained_model\model.safetensors'
        Hash = 'bd7c823ecb72f761cf61bbca76eb945184be4d990c0a18e8f014deff7c22da5b'
    }
)
$auxiliary = @('config.json','policy_preprocessor.json','policy_postprocessor.json',
    'policy_preprocessor_step_5_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors')
$protected = @(
    'data\models\smolvla_base\model.safetensors',
    'outputs\v2c30_sft_overnight_v1\checkpoints\step_0009426_best\pretrained_model\model.safetensors',
    'outputs\gripper_control_fix_20261019\precision_repair\training\expert\expert.safetensors',
    'outputs\five_episode_openhold_20261020\training\expert\expert.safetensors',
    'outputs\fp32_multiepisode_20261020\training\checkpoints\step_0011300_final\pretrained_model\model.safetensors'
)
$protectedHashes = @{}
foreach ($relative in $protected) {
    $file = Resolve-WorkspaceFile $relative
    $protectedHashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
}
$records = @()
foreach ($pair in $pairs) {
    $target = Resolve-WorkspaceFile $pair.Remove
    $retained = Resolve-WorkspaceFile $pair.Keep
    if ($target -eq $retained -or -not $target.StartsWith($outputRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Invalid exact cleanup target'
    }
    if ((Split-Path -Leaf $target) -ne 'model.safetensors' -or $protectedHashes.ContainsKey($target)) {
        throw 'Protected or unexpected target'
    }
    foreach ($file in @($target,$retained)) {
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pair.Hash) {
            throw 'Model hash differs from verified duplicate pair'
        }
    }
    $auxHashes = @{}
    foreach ($name in $auxiliary) {
        $old = Join-Path (Split-Path -Parent $target) $name
        $new = Join-Path (Split-Path -Parent $retained) $name
        $oldHash = (Get-FileHash -LiteralPath $old -Algorithm SHA256).Hash
        if ($oldHash -ne (Get-FileHash -LiteralPath $new -Algorithm SHA256).Hash) {
            throw 'Configuration or normalization differs'
        }
        $auxHashes[$name] = $oldHash.ToLowerInvariant()
    }
    $records += [ordered]@{target=$target; retained_identical_copy=$retained; sha256=$pair.Hash;
        bytes=(Get-Item -LiteralPath $target).Length; auxiliary_hashes=$auxHashes; removed=$false;
        restore='Copy retained_identical_copy back to target, then verify sha256; no unique model weights are lost.'}
}
$totalBytes = 0L
foreach ($record in $records) { $totalBytes += [long]$record['bytes'] }
if (-not $Execute) {
    [ordered]@{dry_run=$true; bytes_to_reclaim=$totalBytes; records=$records} | ConvertTo-Json -Depth 10
    exit 0
}
$receiptDirectory = Join-Path $outputRoot 'checkpoint_cleanup_20261020'
if (Test-Path -LiteralPath $receiptDirectory) { throw 'Existing cleanup receipt; no automatic retry' }
New-Item -ItemType Directory -Path $receiptDirectory | Out-Null
$disk = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($workspaceRoot))
$receipt = [ordered]@{completed=$false; started_utc=[DateTime]::UtcNow.ToString('o');
    authorization='User requested deletion of unused checkpoints; remove only four proven redundant weight copies.';
    disk_free_before_bytes=$disk.AvailableFreeSpace; records=$records; protected_weights_sha256=$protectedHashes;
    optimizer_states_removed=$false; unique_weights_removed=$false; datasets_logs_videos_removed=$false;
    logical_bytes_removed=0; source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
Write-Receipt (Join-Path $receiptDirectory 'receipt.json') $receipt
foreach ($record in $records) {
    # Resolve and recheck both literal files immediately before each deletion.
    $target = Resolve-WorkspaceFile ($record.target.Substring($workspaceRoot.Length + 1))
    $retained = Resolve-WorkspaceFile ($record.retained_identical_copy.Substring($workspaceRoot.Length + 1))
    foreach ($file in @($target,$retained)) {
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw 'File changed' }
    }
    $marker = Join-Path (Split-Path -Parent $target) 'RETIRED_DUPLICATE_MODEL.json'
    Write-Receipt $marker $record
    Remove-Item -LiteralPath $target -Force
    if (Test-Path -LiteralPath $target) { throw 'Weight deletion did not complete' }
    if ((Get-FileHash -LiteralPath $retained -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw 'Retained copy changed' }
    $record.removed = $true
    $receipt.logical_bytes_removed += $record.bytes
    Write-Receipt $marker $record
    Write-Receipt (Join-Path $receiptDirectory 'receipt.json') $receipt
}
foreach ($file in $protectedHashes.Keys) {
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $protectedHashes[$file]) {
        throw 'Protected weights changed'
    }
}
$receipt.completed = $true
$receipt.ended_utc = [DateTime]::UtcNow.ToString('o')
$receipt.disk_free_after_bytes = (New-Object IO.DriveInfo([IO.Path]::GetPathRoot($workspaceRoot))).AvailableFreeSpace
Write-Receipt (Join-Path $receiptDirectory 'receipt.json') $receipt
[ordered]@{completed=$true; weight_copies_removed=$records.Count;
    logical_gib_reclaimed=$receipt.logical_bytes_removed/1GB;
    disk_free_before_gib=$receipt.disk_free_before_bytes/1GB;
    disk_free_after_gib=$receipt.disk_free_after_bytes/1GB;
    receipt=(Join-Path $receiptDirectory 'receipt.json')} | ConvertTo-Json
