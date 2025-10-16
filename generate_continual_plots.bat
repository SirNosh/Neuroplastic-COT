@echo off
REM Generate all continual learning visualizations

echo ========================================
echo Generating Continual Learning Plots
echo ========================================
echo.

REM Check if results exist
if not exist "output_continual_baseline\continual_results.json" (
    echo ERROR: Baseline results not found!
    echo Please run run_continual_experiments.bat first
    pause
    exit /b 1
)

REM Create plots directory
if not exist "plots_continual" mkdir plots_continual

REM Generate all visualizations
python plot_continual.py ^
    --results_files ^
        ./output_continual_baseline/continual_results.json ^
        ./output_continual_ewc/continual_results.json ^
        ./output_continual_si/continual_results.json ^
        ./output_continual_full/continual_results.json ^
    --model_names ^
        baseline ^
        ewc ^
        si ^
        full_neuroplastic ^
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
pause
