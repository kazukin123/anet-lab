@echo off
title %~nx0
setlocal
set "TOOL_EXE=%~dp0..\..\.venv\Scripts\mlflow.exe"
if not exist "%TOOL_EXE%" (
    echo [ERROR] Python tool was not found: "%TOOL_EXE%"
    exit /b 1
)
start "" "%TOOL_EXE%" server
sleep 10
start "" "http://localhost:5000/"
