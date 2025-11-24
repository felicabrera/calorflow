# Run a sample train local (Preprocess + Training) for FCC process using PowerShell
$env:PYTHONPATH = '.'
Write-Host "Calling /api/preprocess..."
$pre = Invoke-RestMethod -Uri http://localhost:8000/api/preprocess -Method Post
Write-Host "Preprocess Response:`n$($pre | ConvertTo-Json -Depth 5)"
Write-Host "Starting training (FCC)..."
$t = Invoke-RestMethod -Uri http://localhost:8000/api/train -Method Post -Body (@{process = 'FCC'; n_trials = 4; use_optuna = $false } | ConvertTo-Json) -ContentType 'application/json'
Write-Host "Train Response:`n$($t | ConvertTo-Json -Depth 5)"
