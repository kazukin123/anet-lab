@echo off
setlocal

rem === batファイルの場所を基準にパスを構築 ===
set ROOT=%~dp0..\..
rem cd /d "%ROOT%"

set CARTPOLE_EXE=cart2\bin\release\CartPole2.exe
set VIEWER_JAR=..\..\viewers\metrics-viewer\target\metrics-viewer.jar
set VIEWER_PORT=8082
pwd

echo.
echo [1/2] Starting Metrics Viewer (port %VIEWER_PORT%)...
start "MetricsViewer" cmd /c ^
   "java -jar %VIEWER_JAR% --server.port=%VIEWER_PORT%"

rem === Spring Boot起動待機（Tomcat初期化） ===
timeout /t 5 /nobreak >nul

echo.
echo [2/2] Opening browser...
start "" "http://localhost:%VIEWER_PORT%"

echo.
echo All processes launched. Press any key to close this window.
rem pause >nul
endlocal
exit /b
