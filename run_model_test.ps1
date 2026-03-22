$projectDir = $PSScriptRoot

$body = @{
  GOOGLE_PAID_SEARCH_SPEND = 1000
  GOOGLE_SHOPPING_SPEND = 800
  GOOGLE_PMAX_SPEND = 300
  META_FACEBOOK_SPEND = 900
  META_INSTAGRAM_SPEND = 200
  EMAIL_CLICKS = 500
  ORGANIC_SEARCH_CLICKS = 1200
  DIRECT_CLICKS = 700
  BRANDED_SEARCH_CLICKS = 350
  year = 2024
  month = 5
  day_of_week = 2
} | ConvertTo-Json

try {
  $health = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/' -Method Get
  Write-Host "Health check:" $health
} catch {
  Write-Error "Cannot reach http://127.0.0.1:5000/. Start mmm_app.py first."
  exit 1
}

try {
  $result = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/predict' -Method Post -ContentType 'application/json' -Body $body
  Write-Host ""
  Write-Host "Prediction result:"
  $result | Format-List
} catch {
  Write-Error "Prediction request failed: $($_.Exception.Message)"
  exit 1
}
