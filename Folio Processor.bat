@echo off
rem Double-click to open the Folio Processor desktop app.
rem Portable: tries the py launcher, then python on PATH. No hard-coded paths.
where pyw >nul 2>&1 && (start "" pyw -m folio.gui & exit /b)
where pythonw >nul 2>&1 && (start "" pythonw -m folio.gui & exit /b)
py -m folio.gui 2>nul || python -m folio.gui
