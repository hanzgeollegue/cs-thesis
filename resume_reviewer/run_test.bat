@echo off
echo Activating virtual environment...
call ..\venv311\Scripts\activate

echo.
echo Running robust scoring pipeline tests...
echo ========================================
python test_robust_scoring.py

echo.
echo Test completed. Press any key to exit...
pause > nul
