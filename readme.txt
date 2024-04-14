Real time strategy 
Broker - Oanda

1. install redis first and run the redis server:
for windows, get from https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip 
download it, unzip it, put it anywhere
click the redis-server.exe to run the server

2. run the following commands in terminals for env:
conda create --name 5010_env python=3.11
conda activate 5010_env
pip install -r requirement.txt

3. the model is pretrained, so you can run the following command to see the result:
init the backend first : python main.py
run the frontend: python frontend.py

4. optional: if you want to train the model, you can run the following command:
python train.py
