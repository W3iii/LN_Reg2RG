# 監督式重跑: build_lobe_masks.py 連續失敗會 exit 2, 這裡用全新 process 續跑
# (CUDA context 一旦壞掉, 同一個 process 內無法復原, 只能重開)
$py  = "$env:USERPROFILE\miniconda3\envs\lobeseg\python.exe"
$scr = "d:\W3iii\NCKU\DataSet\reg2rg_dataset\build_lobe_masks.py"
$dir = "d:\W3iii\NCKU\DataSet\reg2rg_dataset\lobe"

for ($i = 1; $i -le 30; $i++) {
    $done = (Get-ChildItem "$dir\*_lobe5.npz" -ErrorAction SilentlyContinue).Count
    Write-Output "=== attempt $i | 已完成 $done / 1400 ==="
    & $py $scr 2>&1 | Select-String -Pattern "^\[|^ok=|^== "
    if ($LASTEXITCODE -eq 0) {
        Write-Output "=== 全部完成 ==="
        break
    }
    Write-Output "=== exit $LASTEXITCODE, 60 秒後以新 process 重試 ==="
    Start-Sleep -Seconds 60
}
