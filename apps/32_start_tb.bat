cd /d "%~dp0runner"

start tensorboard --logdir runs
sleep 10
start "" "http://localhost:6006/"
