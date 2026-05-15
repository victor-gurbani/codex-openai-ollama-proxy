@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not "%~1"=="" set "PORT=%~1"
if not "%~2"=="" set "CODEX_AUTH_PATH=%~2"

if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run install-proxy.bat first.
    exit /b 1
)

echo Starting codex-openai-ollama-proxy
echo Working directory: %CD%
echo Config sources: CLI args ^> environment variables ^> .env ^> built-in defaults
call :print_flag DEBUG
call :print_flag DISABLE_COPILOT_ADAPTATIONS
echo.

"%SCRIPT_DIR%.venv\Scripts\python.exe" -m codex_openai_ollama_proxy

endlocal
exit /b %ERRORLEVEL%

:print_flag
set "FLAG_NAME=%~1"
call set "FLAG_VALUE=%%%FLAG_NAME%%%"
if "%FLAG_VALUE%"=="" (
    echo %FLAG_NAME%: not set in shell; Python may load it from .env/defaults
    exit /b 0
)
if /I "%FLAG_VALUE%"=="1" goto flag_on
if /I "%FLAG_VALUE%"=="true" goto flag_on
if /I "%FLAG_VALUE%"=="yes" goto flag_on
if /I "%FLAG_VALUE%"=="on" goto flag_on
echo %FLAG_NAME%: off in shell ^('%FLAG_VALUE%'^); passed to Python
exit /b 0

:flag_on
echo %FLAG_NAME%: on in shell; passed to Python
exit /b 0
