param(
    [string]$PipelineRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$AlphaPoseRoot = "F:\LLM-pepper\AlphaPose",
    [string]$FromW1Root = "F:\LLM-pepper\FRoM-W1",
    [string]$MotionId = "point_left_001",
    [string]$SourceVideoName = "video_test_001.mp4"
)

$ErrorActionPreference = "Stop"

function Copy-PathIfExists {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Host "[SKIP] Missing: $Source"
        return
    }
    $parent = Split-Path -Parent $Destination
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    Write-Host "[COPY] $Source -> $Destination"
}

function Copy-MatchingFiles {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$DestinationDir
    )
    if (-not (Test-Path -LiteralPath $SourceDir)) {
        Write-Host "[SKIP] Missing dir: $SourceDir"
        return
    }
    $files = Get-ChildItem -LiteralPath $SourceDir -Filter $Pattern -File
    if (-not $files) {
        Write-Host "[SKIP] No match: $SourceDir\$Pattern"
        return
    }
    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
    foreach ($file in $files) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $DestinationDir $file.Name) -Force
        Write-Host "[COPY] $($file.FullName) -> $DestinationDir"
    }
}

$privateRoot = Join-Path $PipelineRoot "Private"
New-Item -ItemType Directory -Force -Path $privateRoot | Out-Null

# AlphaPose private assets.
Copy-PathIfExists `
    -Source (Join-Path $AlphaPoseRoot "pretrained_models") `
    -Destination (Join-Path $privateRoot "alphapose\pretrained_models")
Copy-MatchingFiles `
    -SourceDir (Join-Path $AlphaPoseRoot "model_files") `
    -Pattern "basicModel_*.pkl" `
    -DestinationDir (Join-Path $privateRoot "alphapose\model_files")
Copy-PathIfExists `
    -Source (Join-Path $AlphaPoseRoot "detector\yolo\data") `
    -Destination (Join-Path $privateRoot "alphapose\detector\yolo\data")
Copy-PathIfExists `
    -Source (Join-Path $AlphaPoseRoot "detector\yolox\data") `
    -Destination (Join-Path $privateRoot "alphapose\detector\yolox\data")

# FRoM-W1 private retarget assets.
Copy-PathIfExists `
    -Source (Join-Path $FromW1Root "H-ACT\retarget\models\smpl") `
    -Destination (Join-Path $privateRoot "fromw1\H-ACT\retarget\models\smpl")
Copy-PathIfExists `
    -Source (Join-Path $FromW1Root "H-ACT\retarget\models\mano") `
    -Destination (Join-Path $privateRoot "fromw1\H-ACT\retarget\models\mano")
Copy-PathIfExists `
    -Source (Join-Path $FromW1Root "H-ACT\retarget\assets") `
    -Destination (Join-Path $privateRoot "fromw1\H-ACT\retarget\assets")

# One small demo package. Default is point_left_001.
$demoRoot = Join-Path $privateRoot "demo_data"
Copy-PathIfExists `
    -Source (Join-Path $PipelineRoot "data\videos\$SourceVideoName") `
    -Destination (Join-Path $demoRoot "videos\$SourceVideoName")
Copy-PathIfExists `
    -Source (Join-Path $PipelineRoot "data\alphapose_raw\$MotionId") `
    -Destination (Join-Path $demoRoot "alphapose_raw\$MotionId")

Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\smpl\smpl_raw") -Pattern "$MotionId*_smpl_raw.npy" -DestinationDir (Join-Path $demoRoot "smpl\smpl_raw")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\smpl\smpl_repaired") -Pattern "$MotionId*_smpl_repaired.npy" -DestinationDir (Join-Path $demoRoot "smpl\smpl_repaired")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\smpl\smpl_repaired_compact") -Pattern "$MotionId*_smpl_repaired_compact.npy" -DestinationDir (Join-Path $demoRoot "smpl\smpl_repaired_compact")

Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\canonical\canonical_original") -Pattern "$MotionId*.npz" -DestinationDir (Join-Path $demoRoot "canonical\canonical_original")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\canonical\canonical_edited") -Pattern "$MotionId*.npz" -DestinationDir (Join-Path $demoRoot "canonical\canonical_edited")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\canonical\canonical_fromw1_llm_edited") -Pattern "$MotionId*" -DestinationDir (Join-Path $demoRoot "canonical\canonical_fromw1_llm_edited")

Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\fromw1_inputs\fromw1_inputs_canonical_v2") -Pattern "$MotionId*.npy" -DestinationDir (Join-Path $demoRoot "fromw1_inputs\fromw1_inputs_canonical_v2")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\fromw1_pkl\fromw1_pkl_canonical_v2") -Pattern "$MotionId*.pkl" -DestinationDir (Join-Path $demoRoot "fromw1_pkl\fromw1_pkl_canonical_v2")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\h1_reference\fromw1_pkl_canonical_v2") -Pattern "$MotionId*.npz" -DestinationDir (Join-Path $demoRoot "h1_reference\fromw1_pkl_canonical_v2")

Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\gifs\h1_reference_gifs\fromw1_pkl_canonical_v2") -Pattern "$MotionId*.gif" -DestinationDir (Join-Path $demoRoot "gifs\h1_reference_gifs\fromw1_pkl_canonical_v2")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\gifs\pkl_gifs\fromw1_pkl_canonical_v2") -Pattern "$MotionId*.gif" -DestinationDir (Join-Path $demoRoot "gifs\pkl_gifs\fromw1_pkl_canonical_v2")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\gifs\RoBoJuDo_H1_gifs\robojudo_pkl_canonical_v2_gifs") -Pattern "$MotionId*.gif" -DestinationDir (Join-Path $demoRoot "gifs\RoBoJuDo_H1_gifs\robojudo_pkl_canonical_v2_gifs")

Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\metrics\alphapose_quality") -Pattern "$MotionId*" -DestinationDir (Join-Path $demoRoot "metrics\alphapose_quality")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\metrics\canonical_edited_quality") -Pattern "$MotionId*" -DestinationDir (Join-Path $demoRoot "metrics\canonical_edited_quality")
Copy-MatchingFiles -SourceDir (Join-Path $PipelineRoot "data\metrics\execution_robojudo_fromw1_pkl_canonical_v2") -Pattern "$MotionId*" -DestinationDir (Join-Path $demoRoot "metrics\execution_robojudo_fromw1_pkl_canonical_v2")

Copy-PathIfExists `
    -Source (Join-Path $PipelineRoot "data\llm\llm_visual_edit\qwen_vl_h1_reference") `
    -Destination (Join-Path $demoRoot "llm\llm_visual_edit\qwen_vl_h1_reference")
Copy-PathIfExists `
    -Source (Join-Path $PipelineRoot "data\llm\llm_edits\qwen_max_h1_reference") `
    -Destination (Join-Path $demoRoot "llm\llm_edits\qwen_max_h1_reference")

Write-Host ""
Write-Host "[DONE] Private package prepared at: $privateRoot"
Write-Host "[NOTE] API keys are intentionally not copied. Do not add qw_LLM.txt or .env to this folder."
