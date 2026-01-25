@echo off
:: SecuRAG Build and Deployment Script
:: CONFIANZA23 SECURE-SDLC-001 Compliant

echo [*] Initializing SecuRAG environment...

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python 3.12+ is required but not found.
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist "venv" (
    echo [*] Creating virtual environment...
    python -m venv venv
)

:: 3. Install Dependencies
echo [*] Upgrading pip to ensure wheel support...
call venv\Scripts\activate
python -m pip install --upgrade pip

echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt

:: 4. Pull Ollama Models
echo [*] Checking for Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama is not installed or not in PATH.
    echo [!] Please install Ollama from https://ollama.com/ and ensure it is running.
    echo [!] Skipping model pull...
) else (
    echo [*] Pulling required Ollama models...
    ollama pull llama3
    ollama pull nomic-embed-text
)

:: 5. Setup Docs Directory
if not exist "docs" (
    echo [*] Creating docs directory...
    mkdir docs
)

echo.
echo [OK] SecuRAG project environment is configured.
echo [!] IMPORTANT: Ollama must be RUNNING for this system to work.
echo [!] If model pull failed, please run these manually AFTER installing Ollama:
echo     ollama pull llama3
echo     ollama pull nomic-embed-text
echo.
echo [*] To run the system, use: python main.py --dir docs
pause
