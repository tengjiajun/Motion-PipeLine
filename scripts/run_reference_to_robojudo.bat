@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM One-click pipeline:
REM canonical_motion.npz -> FRoM-W1 623 npy -> FRoM-W1 H1 pkl
REM -> reference pkl GIF -> RoboJuDo rollout data + execution GIF.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PIPELINE_ROOT=%%~fI"

if "%~1"=="" (
  echo Usage:
  echo   %~nx0 ^<canonical_motion.npz^> [run_name]
  echo.
  echo Example:
  echo   %~nx0 data\canonical\canonical_llm_edited\point_left_001_canonical_motion_llm_v3.npz point_left_001_llm_v3
  exit /b 2
)

set "INPUT_NPZ=%~1"
if not exist "%INPUT_NPZ%" (
  if exist "%PIPELINE_ROOT%\%INPUT_NPZ%" (
    set "INPUT_NPZ=%PIPELINE_ROOT%\%INPUT_NPZ%"
  ) else (
    echo [ERROR] Input canonical npz not found: %~1
    exit /b 2
  )
)
for %%I in ("%INPUT_NPZ%") do set "INPUT_NPZ=%%~fI"

if "%~2"=="" (
  for %%I in ("%INPUT_NPZ%") do set "RUN_NAME=%%~nI"
) else (
  set "RUN_NAME=%~2"
)

REM Override these from the shell if your environment changes.
if "%PIPELINE_PYTHON%"=="" set "PIPELINE_PYTHON=python"
if "%RETARGET_PYTHON%"=="" set "RETARGET_PYTHON=F:\anaconda\envs\retarget\python.exe"
if "%ROBOJUDO_PYTHON%"=="" set "ROBOJUDO_PYTHON=F:\anaconda\envs\robojudo\python.exe"
if "%FROMW1_ROOT%"=="" set "FROMW1_ROOT=F:\LLM-pepper\FRoM-W1"
if "%FROMW1_RETARGET_ROOT%"=="" set "FROMW1_RETARGET_ROOT=%FROMW1_ROOT%\H-ACT\retarget"
if "%H1_XML%"=="" set "H1_XML=%FROMW1_RETARGET_ROOT%\assets\robot\h1\h1.xml"

set "FROMW1_NPY_DIR=%PIPELINE_ROOT%\data\fromw1_inputs\fromw1_inputs_llm_runs\%RUN_NAME%"
set "FROMW1_PKL_DIR=%PIPELINE_ROOT%\data\fromw1_pkl\fromw1_pkl_llm_runs\%RUN_NAME%"
set "PKL_GIF_DIR=%PIPELINE_ROOT%\data\gifs\pkl_gifs\%RUN_NAME%"
set "ROBOJUDO_OUT_DIR=%PIPELINE_ROOT%\data\results\robojudo\%RUN_NAME%"

echo [INFO] Pipeline root: %PIPELINE_ROOT%
echo [INFO] Input npz:     %INPUT_NPZ%
echo [INFO] Run name:      %RUN_NAME%
echo [INFO] FRoM-W1 root:  %FROMW1_ROOT%
echo.

pushd "%PIPELINE_ROOT%" || exit /b 1

echo [1/5] canonical npz -^> FRoM-W1 623 npy
"%PIPELINE_PYTHON%" adapters\canonical_to_fromw1_623.py ^
  --input-npz "%INPUT_NPZ%" ^
  --output-dir "%FROMW1_NPY_DIR%" ^
  --fromw1-root "%FROMW1_ROOT%"
if errorlevel 1 goto :fail

echo.
echo [2/5] FRoM-W1 623 npy -^> H1 pkl
pushd "%FROMW1_RETARGET_ROOT%" || goto :fail
"%RETARGET_PYTHON%" "%PIPELINE_ROOT%\tools\fromw1_623_to_pkl_batch.py" ^
  --input-dir "%FROMW1_NPY_DIR%" ^
  --output-dir "%FROMW1_PKL_DIR%" ^
  --fromw1-retarget-root "%FROMW1_RETARGET_ROOT%" ^
  --robot H1 ^
  --hand-type dex3 ^
  --output-fps 60
if errorlevel 1 goto :fail_pop
popd

echo.
echo [3/5] H1 pkl -^> reference GIF
if not exist "%PKL_GIF_DIR%" mkdir "%PKL_GIF_DIR%"
pushd "%FROMW1_RETARGET_ROOT%" || goto :fail
for %%P in ("%FROMW1_PKL_DIR%\*.pkl") do (
  echo [GIF] %%~nxP
  "%RETARGET_PYTHON%" scripts\pkl_2_gif.py ^
    --input "%%~fP" ^
    --xml "%H1_XML%" ^
    --output "%PKL_GIF_DIR%\%%~nP_h1.gif" ^
    --width 640 ^
    --height 480 ^
    --fps 30 ^
    --step 1 ^
    --exposure 2.2 ^
    --gamma 0.8 ^
    --camera-azimuth 180 ^
    --camera-elevation -6 ^
    --camera-min-distance 3.4 ^
    --camera-lookat-height 0.72 ^
    --auto-fit ^
    --auto-fit-margin 0.22
  if errorlevel 1 goto :fail_pop
)
popd

echo.
echo [4/5] H1 pkl -^> RoboJuDo rollout data + execution GIF
"%ROBOJUDO_PYTHON%" tools\robojudo_h1_pkl_batch_export.py ^
  --input-dir "%FROMW1_PKL_DIR%" ^
  --output-dir "%ROBOJUDO_OUT_DIR%" ^
  --width 640 ^
  --height 360 ^
  --gif-fps 25 ^
  --capture-every-n-steps 2
if errorlevel 1 goto :fail

echo.
echo [5/5] Done
echo [OUT] FRoM-W1 npy:   %FROMW1_NPY_DIR%
echo [OUT] FRoM-W1 pkl:   %FROMW1_PKL_DIR%
echo [OUT] pkl GIFs:      %PKL_GIF_DIR%
echo [OUT] RoboJuDo data: %ROBOJUDO_OUT_DIR%
popd
exit /b 0

:fail_pop
popd
:fail
echo.
echo [ERROR] Pipeline failed.
popd
exit /b 1
