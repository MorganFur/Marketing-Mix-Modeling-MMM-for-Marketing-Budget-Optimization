$projectDir = $PSScriptRoot
$scriptPath = Join-Path $projectDir 'evaluate_model.py'

if (-not (Test-Path -LiteralPath $scriptPath)) {
  Write-Error "Evaluation script not found: $scriptPath"
  exit 1
}

$cutoffDate = if ($args.Count -ge 1) { $args[0] } else { '2024-01-01' }

Push-Location $projectDir
try {
  python .\evaluate_model.py $cutoffDate
} finally {
  Pop-Location
}
