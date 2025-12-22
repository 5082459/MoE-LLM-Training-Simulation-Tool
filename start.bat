@echo off
setlocal

echo 🚀 Starting AI Infra Simulation Tool...

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment (if not exists)
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
) else (
    echo ✅ Virtual environment detected.
)

:: 3. Activate Virtual Environment
call venv\Scripts\activate

:: 4. Install Dependencies
if exist "requirements.txt" (
    echo ⬇️  Installing/Updating dependencies...
    pip install -r requirements.txt -q
    echo ✅ Dependencies installed.
) else (
    echo ⚠️ Warning: requirements.txt not found.
)

:: 5. Start Services
echo 🔥 Starting services...

:: Start Backend (in a new minimized window)
echo    - Starting Backend API (Port 8000)...
start /min "AI Infra Backend" cmd /c "uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4"

:: Wait a bit for backend to start
timeout /t 3 /nobreak >nul

:: Start Frontend
echo    - Starting Frontend UI...
echo ✅ Services ready! Browser should open automatically.
echo 🛑 Close this window to stop the Frontend.
echo    (Note: Close the "AI Infra Backend" window manually to stop the backend)

streamlit run frontend_app.py

pause
