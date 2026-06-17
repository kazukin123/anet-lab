@echo off
setlocal

rem === batファイルの場所を基準にパスを構築 ===
set "ROOT=%~dp0..\.."
set "VIEWER_JAR=%ROOT%\viewers\metrics-viewer\target\metrics-viewer.jar"
set "RUNS_DIR=%ROOT%\apps\runner\runs_optuna"
set "VIEWER_PORT=8083"
pwd

echo.
echo [1/2] Starting Metrics Viewer (port %VIEWER_PORT%)...
start "MetricsViewer" cmd /c ^
   "java -Xverify:none -jar "%VIEWER_JAR%" --server.port=%VIEWER_PORT% --metricsviewer.runs-dir="%RUNS_DIR%""

rem === Spring Boot起動待機（Tomcat初期化） ===
timeout /t 20 /nobreak >nul

echo.
echo [2/2] Opening browser...
start "" "http://localhost:%VIEWER_PORT%"

echo.
echo All processes launched. Press any key to close this window.
rem pause >nul
endlocal
exit /b
