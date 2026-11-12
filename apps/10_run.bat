@echo off
cd /d "%~dp0runner"
REM SET EXE="bin\RelWithDebInfo\AnetRLRunner.exe"
SET EXE="bin\Release\AnetRLRunner.exe"

%EXE% %*

