# Automated FL round runner — 5 rounds x 3 hospitals
# Run from: D:\tmp\fedlearn-fabric
# Requires: FL REST Server + Fabric network + IPFS already running

$ROOT      = "D:\tmp\fedlearn-fabric"
$CLIENT    = "$ROOT\client\fl_client.py"
$METRICS   = "$ROOT\server\metrics.json"
$ROUNDS    = 5
$HOSPITALS = @("Hospital1", "Hospital2", "Hospital3")

function Submit-Hospital($hospital, $round) {
    Write-Host "  → $hospital  Round $round" -ForegroundColor Yellow
    python $CLIENT `
        --sender   $hospital `
        --model    covid `
        --round    $round `
        --algo     fedprox `
        --mu       0.01 `
        --samples  200 `
        --epochs   2 `
        --lr       0.01
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: $hospital Round $round failed — stopping." -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ $hospital done" -ForegroundColor Green
}

# Clear metrics before starting
Set-Content -Path $METRICS -Value "[]" -Encoding utf8
Write-Host "Metrics cleared." -ForegroundColor Cyan

for ($r = 1; $r -le $ROUNDS; $r++) {
    Write-Host "`n══════════════════════════════" -ForegroundColor Cyan
    Write-Host "  ROUND $r / $ROUNDS" -ForegroundColor Cyan
    Write-Host "══════════════════════════════" -ForegroundColor Cyan

    foreach ($h in $HOSPITALS) {
        Submit-Hospital $h $r
    }

    Write-Host "`n  Round $r aggregated. Waiting 3s..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 3

    # Print current accuracy
    $m = Get-Content $METRICS | ConvertFrom-Json
    $latest = $m | Where-Object { $_.round -eq $r } | Select-Object -Last 1
    if ($latest) {
        $pct = [math]::Round($latest.accuracy * 100, 2)
        Write-Host "  Global accuracy Round $r : $pct%" -ForegroundColor Magenta
    }
}

Write-Host "`n══════════════════════════════" -ForegroundColor Green
Write-Host "  ALL $ROUNDS ROUNDS COMPLETE" -ForegroundColor Green
Write-Host "══════════════════════════════" -ForegroundColor Green

$all = Get-Content $METRICS | ConvertFrom-Json
Write-Host "`nFinal Results:"
foreach ($m in $all) {
    $pct = [math]::Round($m.accuracy * 100, 2)
    Write-Host "  Round $($m.round): $pct%"
}
