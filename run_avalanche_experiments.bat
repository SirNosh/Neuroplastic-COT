@echo off
REM Run Avalanche-based continual learning experiments
REM Compares Avalanche library implementations with our custom implementations

echo ========================================
echo Avalanche Continual Learning Experiments
echo ========================================
echo.
echo This will train models using Avalanche library:
echo   1. Naive (baseline - no continual learning)
echo   2. Avalanche-EWC (library implementation)
echo   3. Avalanche-SI (library implementation)
echo.
echo Training phases: Arithmetic -^> Algebra -^> Geometry
echo.
pause

REM Check CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo.

REM =============================================================================
REM 1. NAIVE BASELINE (Avalanche)
REM =============================================================================

echo ========================================
echo 1. Training: Naive Baseline
echo ========================================
echo Strategy: Sequential fine-tuning without continual learning
echo Expected: High catastrophic forgetting (~30-50%%)
echo.

python train_avalanche.py ^
    --strategy naive ^
    --output_dir ./output_avalanche_naive ^
    --num_epochs 2 ^
    --learning_rate 2e-5 ^
    --max_samples_per_phase 1000 ^
    --seed 42

if errorlevel 1 (
    echo ERROR: Naive baseline training failed!
    pause
    exit /b 1
)

echo.
echo Naive baseline complete!
echo.

REM =============================================================================
REM 2. AVALANCHE EWC
REM =============================================================================

echo ========================================
echo 2. Training: Avalanche EWC
echo ========================================
echo Strategy: Elastic Weight Consolidation (Avalanche library)
echo Expected: Low catastrophic forgetting (~10-15%%)
echo.

python train_avalanche.py ^
    --strategy ewc ^
    --ewc_lambda 0.4 ^
    --output_dir ./output_avalanche_ewc ^
    --num_epochs 2 ^
    --learning_rate 2e-5 ^
    --max_samples_per_phase 1000 ^
    --seed 42

if errorlevel 1 (
    echo ERROR: Avalanche EWC training failed!
    pause
    exit /b 1
)

echo.
echo Avalanche EWC complete!
echo.

REM =============================================================================
REM 3. AVALANCHE SI
REM =============================================================================

echo ========================================
echo 3. Training: Avalanche SI
echo ========================================
echo Strategy: Synaptic Intelligence (Avalanche library)
echo Expected: Low catastrophic forgetting (~10-15%%)
echo.

python train_avalanche.py ^
    --strategy si ^
    --si_lambda 0.4 ^
    --output_dir ./output_avalanche_si ^
    --num_epochs 2 ^
    --learning_rate 2e-5 ^
    --max_samples_per_phase 1000 ^
    --seed 42

if errorlevel 1 (
    echo ERROR: Avalanche SI training failed!
    pause
    exit /b 1
)

echo.
echo Avalanche SI complete!
echo.

REM =============================================================================
REM SUMMARY
REM =============================================================================

echo.
echo ========================================
echo All Avalanche Experiments Complete!
echo ========================================
echo.
echo Results saved in:
echo   - ./output_avalanche_naive/continual_results.json
echo   - ./output_avalanche_ewc/continual_results.json
echo   - ./output_avalanche_si/continual_results.json
echo.
echo Next step: Generate comparison plots
echo Run: generate_continual_plots.bat
echo.
pause
