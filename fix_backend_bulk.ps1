$f = "d:\2026\20260101\Self\self-Z\src\backend\backend.c"
$c = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::UTF8)
$orig = $c

# 函数调用缺() — 已知模式
$c = $c -replace 'TrainingConfig train_config = training_config_default;', 'TrainingConfig train_config = training_config_default();'
$c = $c -replace 'PlanConfig plan_cfg = planning_get_default_config;', 'PlanConfig plan_cfg = planning_get_default_config();'
$c = $c -replace 'SDEStereoConfig stereo_cfg = sde_stereo_get_default_config;', 'SDEStereoConfig stereo_cfg = sde_stereo_get_default_config();'

# 数组缺[] — 已知的特定模式
$c = $c -replace 'const char\* perm_names = \{"只读"', 'const char* perm_names[] = {"只读"'

# GpuBackend数组
$c = $c -replace 'GpuBackend all_backends = \{', 'GpuBackend all_backends[] = {'

# 更多const char*数组 (通过读取剩余的特定行)
if ($c -ne $orig) {
    $u = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText($f, $c, $u)
    Write-Host "Fixed"
} else {
    Write-Host "No changes"
}
