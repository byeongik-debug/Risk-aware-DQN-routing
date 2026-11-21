$env:KMP_DUPLICATE_LIB_OK="TRUE"
Set-Location $PSScriptRoot

$factors = @("util", "cong", "delay_inst", "loss_inst", "link_type")

foreach ($f in $factors) {
    Write-Host "===== [$f] run1 시작 =====" -ForegroundColor Cyan
    python train_factor_ablation.py $f 1
    Write-Host "===== [$f] run1 완료 =====" -ForegroundColor Green
}

Write-Host "모든 factor run1 완료" -ForegroundColor Yellow