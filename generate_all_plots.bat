@echo off
REM Generate comprehensive continual learning visualizations
REM Includes both custom implementations and Avalanche library results

echo ========================================
echo Generating Continual Learning Plots
echo ========================================
echo.

REM Create plots directory
if not exist "plots_continual" mkdir plots_continual

echo Checking which experiments have completed...
echo.

REM Initialize result files array
set RESULT_FILES=
set MODEL_NAMES=

REM Check custom implementations
if exist "output_continual_baseline\continual_results.json" (
    echo [FOUND] Custom Baseline
    set RESULT_FILES=%RESULT_FILES% ./output_continual_baseline/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% baseline
)

if exist "output_continual_ewc\continual_results.json" (
    echo [FOUND] Custom EWC
    set RESULT_FILES=%RESULT_FILES% ./output_continual_ewc/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% custom_ewc
)

if exist "output_continual_si\continual_results.json" (
    echo [FOUND] Custom SI
    set RESULT_FILES=%RESULT_FILES% ./output_continual_si/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% custom_si
)

if exist "output_continual_full\continual_results.json" (
    echo [FOUND] Custom Full Neuroplastic
    set RESULT_FILES=%RESULT_FILES% ./output_continual_full/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% custom_full
)

REM Check Avalanche implementations
if exist "output_avalanche_naive\continual_results.json" (
    echo [FOUND] Avalanche Naive
    set RESULT_FILES=%RESULT_FILES% ./output_avalanche_naive/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% avalanche_naive
)

if exist "output_avalanche_ewc\continual_results.json" (
    echo [FOUND] Avalanche EWC
    set RESULT_FILES=%RESULT_FILES% ./output_avalanche_ewc/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% avalanche_ewc
)

if exist "output_avalanche_si\continual_results.json" (
    echo [FOUND] Avalanche SI
    set RESULT_FILES=%RESULT_FILES% ./output_avalanche_si/continual_results.json
    set MODEL_NAMES=%MODEL_NAMES% avalanche_si
)

echo.

REM Check if any results exist
if "%RESULT_FILES%"=="" (
    echo ERROR: No experiment results found!
    echo Please run experiments first:
    echo   - run_continual_experiments.bat ^(custom implementations^)
    echo   - run_avalanche_experiments.bat ^(Avalanche library^)
    pause
    exit /b 1
)

REM Generate all visualizations
echo Generating plots for all available experiments...
echo.

python plot_continual.py ^
    --results_files %RESULT_FILES% ^
    --model_names %MODEL_NAMES% ^
    --output_dir ./plots_continual

if errorlevel 1 (
    echo ERROR: Plot generation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Plots Generated Successfully!
echo ========================================
echo.
echo Visualizations saved in: ./plots_continual/
echo.
echo Generated plots:
echo   - forgetting_curves.png      : Performance retention across phases
echo   - performance_matrices.png   : Task performance after each phase
echo   - forgetting_heatmaps.png    : Forgetting percentages per model
echo   - forgetting_comparison.png  : Average forgetting comparison
echo.
echo Comparison includes:
echo   - Custom implementations (EWC, SI)
echo   - Avalanche library implementations
echo   - Baseline models (no continual learning)
echo.
pause
