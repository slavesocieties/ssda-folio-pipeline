@echo off
rem Double-click to set up (first time) and open the Folio Processor web app.
rem Reuses your installed Python; installs missing pieces + weights on first run.
cd /d "%~dp0"
where py >nul 2>&1 && (py run_app.py & goto done)
python run_app.py
:done
echo.
pause
