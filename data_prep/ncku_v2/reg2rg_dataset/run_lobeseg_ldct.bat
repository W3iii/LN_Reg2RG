@echo off
setlocal
REM ==== LDCT_test lobe segmentation (TotalSegmentator) - double-click to run ====
REM Creates %USERPROFILE%\lobeseg_venv on first run (PyTorch CUDA + TotalSegmentator),
REM checks the GPU, runs a 3-patient smoke test, then the full batch (615 patients with nodules).
REM Run with argument --all to segment all 1000 patients.
set "HERE=%~dp0"
set "VENV=%USERPROFILE%\lobeseg_venv"
set "PY=%VENV%\Scripts\python.exe"
set "BASEPY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "SETUPLOG=%HERE%lobe_setup_log_ldct.txt"
set /a TRIES=0

if exist "%PY%" goto check_gpu
echo [1/5] Creating Python environment at %VENV% ...
"%BASEPY%" -m venv "%VENV%" >> "%SETUPLOG%" 2>&1 || goto fail
echo [2/5] Installing PyTorch with CUDA 12.6 - about 3 GB, this takes a while ...
"%PY%" -m pip install --upgrade pip >> "%SETUPLOG%" 2>&1
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 >> "%SETUPLOG%" 2>&1 || goto fail
echo [3/5] Installing TotalSegmentator ...
"%PY%" -m pip install TotalSegmentator nibabel >> "%SETUPLOG%" 2>&1 || goto fail

:check_gpu
echo [4/5] Checking GPU ...
"%PY%" -c "import torch;assert torch.cuda.is_available(),'CUDA not available';print('GPU OK:',torch.cuda.get_device_name(0),'| torch',torch.__version__)" > "%HERE%gpu_check_ldct.txt" 2>&1
type "%HERE%gpu_check_ldct.txt"
findstr /C:"GPU OK" "%HERE%gpu_check_ldct.txt" >nul || goto fail

echo [5/5] Smoke test on 3 patients - first run also downloads the model weights ...
"%PY%" "%HERE%build_lobe_masks_ldct.py" --limit 3 || goto fail
echo.
echo Smoke test OK. Starting full run - about 5 to 6 hours for 615 patients. Keep this window open.
echo Progress is written to %HERE%lobe_seg_log_ldct.txt
echo.

:loop
set /a TRIES+=1
"%PY%" "%HERE%build_lobe_masks_ldct.py" %*
if %errorlevel%==0 goto done
if %TRIES% GEQ 30 goto fail
echo Batch stopped with code %errorlevel% - restarting in 60 seconds, attempt %TRIES%
timeout /t 60 /nobreak >nul
goto loop

:done
echo.
echo ==== ALL DONE ====
pause
exit /b 0

:fail
echo.
echo ==== FAILED - see lobe_setup_log_ldct.txt / gpu_check_ldct.txt / lobe_seg_log_ldct.txt in %HERE% ====
pause
exit /b 1
