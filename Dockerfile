# 使用官方 Python 映像 (slim-buster 版本通常包含一些基礎工具，且比 alpine 更容易處理 C 相依性)
FROM python:3.9-slim-buster

# 設定環境變數，防止 Python 寫入 .pyc 檔案並啟用無緩衝輸出
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 安裝 open3d 和其他可能的系統相依性
# libgl1-mesa-glx 是 open3d 常見的圖形相依性
# libglib2.0-0 也是 open3d 可能需要的
# libgomp1 有時 numpy/scipy 或其他科學計算庫需要
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 設定容器內的工作目錄
WORKDIR /app

# 複製 requirements.txt 到工作目錄
COPY requirements.txt .

# 安裝 Python 相依套件
# --no-cache-dir 可以減少映像大小
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案中的所有檔案到工作目錄
# (確保您有一個 .dockerignore 檔案來排除不必要的檔案/目錄)
COPY . .

# 暴露 FastAPI 應用程式運行的端口 (與 uvicorn 命令中的端口一致)
EXPOSE 8000

# 容器啟動時執行的命令
# 使用 python -m uvicorn 來執行，並監聽所有網路介面 (0.0.0.0)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]