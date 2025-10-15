@echo off
REM Complete Continual Learning Experiment Pipeline
REM Tests neuroplasticity mechanisms' ability to prevent catastrophic forgetting

echo ========================================
echo Continual Learning Experiments
echo Testing Neuroplasticity in LLMs
echo ========================================
echo.

REM 1. Baseline: Standard Sequential Training (No Neuroplasticity)
echo [1/4] Running Baseline (No Neuroplasticity)...
echo.
python train_continual.py ^
    --num_epochs_per_phase 2 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --max_samples_per_phase 1000 ^
    --output_dir ./output_continual_baseline

if errorlevel 1 (
    echo ERROR: Baseline training failed!
    pause
    exit /b 1
)
echo Baseline completed!
echo.

REM 2. EWC Only
echo [2/4] Running EWC Only...
echo.
python train_continual.py ^
    --use_ewc ^
    --ewc_lambda 0.4 ^
    --num_epochs_per_phase 2 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --max_samples_per_phase 1000 ^
    --output_dir ./output_continual_ewc

if errorlevel 1 (
    echo ERROR: EWC training failed!
    pause
    exit /b 1
)
echo EWC completed!
echo.

REM 3. Hebbian Only
echo [3/4] Running Hebbian Only...
echo.
python train_continual.py ^
    --use_hebbian ^
    --hebb_lambda 0.01 ^
    --num_epochs_per_phase 2 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --max_samples_per_phase 1000 ^
    --output_dir ./output_continual_hebbian

if errorlevel 1 (
    echo ERROR: Hebbian training failed!
    pause
    exit /b 1
)
echo Hebbian completed!
echo.

REM 4. Full Neuroplastic (EWC + ALR + Hebbian)
echo [4/4] Running Full Neuroplastic...
echo.
python train_continual.py ^
    --use_ewc ^
    --use_alr ^
    --use_hebbian ^
    --ewc_lambda 0.4 ^
    --hebb_lambda 0.01 ^
    --num_epochs_per_phase 2 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --max_samples_per_phase 1000 ^
    --output_dir ./output_continual_full

if errorlevel 1 (
    echo ERROR: Full neuroplastic training failed!
    pause
    exit /b 1
)
echo Full neuroplastic completed!
echo.

echo ========================================
echo All Experiments Completed!
echo ========================================
echo.
echo Results saved in:
echo   - Baseline: ./output_continual_baseline/continual_results.json
echo   - EWC: ./output_continual_ewc/continual_results.json
echo   - Hebbian: ./output_continual_hebbian/continual_results.json
echo   - Full: ./output_continual_full/continual_results.json
echo.
echo Next: Run generate_continual_plots.bat to visualize results
echo.
pause
