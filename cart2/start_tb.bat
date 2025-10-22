#python tools\tb_bridge.py --logdir logs --reset
start tensorboard --logdir logs
sleep 8
start "" "http://localhost:6006/"
