$ErrorActionPreference = 'Stop'

# Run this from the repository root:
# C:\Users\霸道志伟\Desktop\GNN\Refrigerant-Solubility-GNN

$Repo = (Get-Location).Path
$Download = 'D:\浏览器下载'
$V6 = Join-Path $Repo 'results_hfc_all\HFC_all_M0'
$V7A = 'D:\折腾\v7_pilot_gpu_results\v7_shadow_experiment\checkpoints\V7-A'
$V7B = 'D:\折腾\v7_pilot_gpu_results2\v7_shadow_experiment\checkpoints\V7-B'
$Freeze = Join-Path $Repo 'v7_shadow_experiment\final_freeze'

New-Item -ItemType Directory -Force -Path $Freeze | Out-Null

# Bring the four generated artifacts into the repository.
$Artifacts = @(
  'v7_master_benchmark_summary.csv',
  'V7_Final_Diagnostic_Report.md',
  'generate_v7_final_freeze_manifest.py'
)
foreach ($name in $Artifacts) {
  $src = Join-Path $Download $name
  if (-not (Test-Path $src)) { throw "Missing download artifact: $src" }
}
Copy-Item (Join-Path $Download 'v7_master_benchmark_summary.csv') (Join-Path $Freeze 'v7_master_benchmark_summary.csv') -Force
Copy-Item (Join-Path $Download 'V7_Final_Diagnostic_Report.md') (Join-Path $Freeze 'V7_Final_Diagnostic_Report.md') -Force
Copy-Item (Join-Path $Download 'generate_v7_final_freeze_manifest.py') (Join-Path $Freeze 'generate_v7_final_freeze_manifest.py') -Force

# Replace the downloaded script with the explicit-root implementation used for final freeze.
# This avoids heuristic V6 detection and therefore catches HFC_all_M0 correctly.
$Script = Join-Path $Freeze 'generate_v7_final_freeze_manifest.py'

uv run python $Script `
  --v6-dir $V6 `
  --v7a-dir $V7A `
  --v7b-dir $V7B `
  --out-dir $Freeze

$Manifest = Join-Path $Freeze 'checkpoint_hash_manifest.json'
if (-not (Test-Path $Manifest)) { throw "Missing generated manifest: $Manifest" }
$Obj = Get-Content $Manifest -Raw | ConvertFrom-Json
if ($Obj.status -ne 'PASS') {
  Write-Host 'FINAL FREEZE BLOCKED: checkpoint manifest is not PASS.' -ForegroundColor Red
  exit 2
}

# Freeze only after the manifest passes all 15 exact checkpoint checks.
git add v7_shadow_experiment/final_freeze
$changed = git status --short
Write-Host $changed

git commit -m 'chore(v7): freeze final shadow experiment provenance'
git push origin main

Write-Host 'V7 FINAL FREEZE COMPLETE' -ForegroundColor Green
