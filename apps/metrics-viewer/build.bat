@echo off
setlocal

rem === Build Metrics Viewer (target\metrics-viewer.jar) with the Maven Wrapper ===
rem === Requires only JDK 17 or later. Maven is downloaded on the first run.    ===

cd /d "%~dp0"

if defined JAVA_HOME goto :build
where java >nul 2>&1 || (
    echo [ERROR] java was not found. Install JDK 17 or later and set JAVA_HOME and Path.
    set EXITCODE=1
    goto :end
)

:build
call "%~dp0mvnw.cmd" -B clean package -Dmaven.test.skip=true
set EXITCODE=%ERRORLEVEL%

echo.
if "%EXITCODE%"=="0" (
    echo Build succeeded: %~dp0target\metrics-viewer.jar
    echo Start Metrics Viewer with apps\22_metrics_viewer_java.bat
) else (
    echo [ERROR] Build failed. See the messages above.
)

:end
echo.
pause
exit /b %EXITCODE%
