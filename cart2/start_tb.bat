#python tools\tb_bridge.py --input ..\out\build\x64-Debug\cart2\train.jsonl --logdir tb_logs --reset
start tensorboard --logdir tb_logs
start "" "http://localhost:6006/"
pause
