# python project
Start-Process uvicorn.exe -ArgumentList "machine-learning.server.ml_server:app", "--host", "0.0.0.0", "--port", "8001", "--reload" -NoNewWindow

