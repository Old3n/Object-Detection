[CmdletBinding()]
param(
    [string]$VideosDir = "videos",
    [string]$Weights = "yolo26n.pt",
    [string]$PedestrianWeights = "",
    [string]$Device = "0",
    [ValidateSet("fp32", "fp16")]
    [string]$Quantize = "fp16",
    [switch]$Show
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([System.IO.Path]::IsPathRooted($VideosDir)) {
    $videoRoot = $VideosDir
}
else {
    $videoRoot = Join-Path $repoRoot $VideosDir
}

if (-not (Test-Path -LiteralPath $videoRoot -PathType Container)) {
    throw "Video directory not found: $videoRoot"
}

$outputRoot = Join-Path $repoRoot "runs\jaad_results"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$videos = @(
    Get-ChildItem -LiteralPath $videoRoot -File -Filter "video_*.mp4" |
    Where-Object { $_.BaseName -match '^video_\d+$' } |
    Sort-Object { [int]($_.BaseName -replace '^video_', '') }
)

if ($videos.Count -eq 0) {
    throw "No video_XXXX.mp4 clips found in $videoRoot"
}

$completed = 0
$failed = [System.Collections.Generic.List[string]]::new()
$total = $videos.Count
$index = 0

Write-Host "Found $total selected clips in $videoRoot" -ForegroundColor Green
Write-Host "Results will be saved to $outputRoot"

foreach ($video in $videos) {
    $index++
    $stem = $video.BaseName
    $outputBase = Join-Path $outputRoot "\${stem}_risk"

    Write-Host ""
    Write-Host "[$index/$total] $($video.Name)" -ForegroundColor Cyan

    $runArgs = @(
        "RiskVideoBenchmark.py",
        "--source", $video.FullName,
        "--weights", $Weights,
        "--tracker", "bytetrack.yaml",
        "--device", $Device,
        "--quantize", $Quantize,
        "--ui-scale", "1.3",
        "--max-display-objects", "12",
        "--output", "$outputBase.mp4",
        "--csv", "$outputBase.csv",
        "--json", "$outputBase.json"
    )

    if ($PedestrianWeights.Trim().Length -gt 0) {
        $pedestrianPath = $PedestrianWeights
        if (-not [System.IO.Path]::IsPathRooted($pedestrianPath)) {
            $pedestrianPath = Join-Path $repoRoot $pedestrianPath
        }
        $runArgs += @("--pedestrian-weights", $pedestrianPath)
        $runArgs += @("--classes", "2", "5", "7", "9")
    }

    if ($Show) {
        $runArgs += "--show"
    }

    Push-Location $repoRoot
    try {
        & python @runArgs
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -eq 0) {
        $completed++
        Write-Host "SUCCESS: $stem" -ForegroundColor Green
    }
    else {
        $failed.Add("$stem (exit $exitCode)")
        Write-Host "FAILED: $stem (exit $exitCode)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========== JAAD BATCH SUMMARY ==========" -ForegroundColor Cyan
Write-Host "Total:     $total"
Write-Host "Completed: $completed" -ForegroundColor Green
Write-Host "Failed:    $($failed.Count)" -ForegroundColor Red
Write-Host "Results:   $outputRoot"

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed clips:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host " - $_" }
}
