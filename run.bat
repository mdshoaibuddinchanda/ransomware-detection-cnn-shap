@echo off
setlocal

echo Running repository validation...
python scripts\ci_validate.py
if errorlevel 1 (
    echo Validation failed. Stopping.
    exit /b 1
)

if /I "%~1"=="gui" goto gui
if /I "%~1"=="notebook" goto notebook

echo Validation passed. Optional modes: run.bat gui or run.bat notebook
exit /b 0

:gui
echo Launching the legacy GUI explicitly...
python Main.py
exit /b %errorlevel%

:notebook
echo Executing the notebook with the already-installed runtime...
python -m nbconvert --to notebook --execute Ransomware_Paper_Enhancements.ipynb --inplace --ExecutePreprocessor.timeout=-1
exit /b %errorlevel%
