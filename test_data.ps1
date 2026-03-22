$dataPath = Join-Path $PSScriptRoot 'test_data.json'

if (-not (Test-Path -LiteralPath $dataPath)) {
  Write-Error "Data file not found: $dataPath"
  exit 1
}

$jsonBody = Get-Content -LiteralPath $dataPath -Raw
$records = $jsonBody | ConvertFrom-Json

if (-not $records) {
  Write-Error "No records found in test_data.json"
  exit 1
}

try {
  $health = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/' -Method Get
  Write-Host "Health check:" $health
} catch {
  Write-Error "Cannot reach http://127.0.0.1:5000/. Start mmm_app.py first."
  exit 1
}

try {
  $result = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/predict' -Method Post -ContentType 'application/json' -Body $jsonBody
  Write-Host ""
  Write-Host "Sent records:" $records.Count
  Write-Host "Prediction result:"
  $result | ConvertTo-Json -Depth 6
} catch {
  Write-Error "Prediction request failed: $($_.Exception.Message)"
  exit 1
}
