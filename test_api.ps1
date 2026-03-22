$body = @{
  GOOGLE_PAID_SEARCH_SPEND = 24
  GOOGLE_SHOPPING_SPEND = 27
  GOOGLE_PMAX_SPEND = 0
  META_FACEBOOK_SPEND = 39
  META_INSTAGRAM_SPEND = 0
  EMAIL_CLICKS = 54
  ORGANIC_SEARCH_CLICKS = 190
  DIRECT_CLICKS = 127
  BRANDED_SEARCH_CLICKS = 39
  year = 2022
  month = 6
  day_of_week = 3
} | ConvertTo-Json

Write-Host "Checking health endpoint..."
$health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/" -Method Get
Write-Host $health

Write-Host ""
Write-Host "Calling /predict..."
$result = Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body
$result | Format-List
