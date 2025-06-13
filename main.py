from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, conlist
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from typing import List
import tempfile
import shutil
from contextlib import asynccontextmanager

# 從 AIModel.py 匯入自訂物件
# 確保 AIModel.py 與 main.py 在同一目錄，或 AIModel 在 PYTHONPATH 中
from AIModel import quaternion_loss, l2_normalize_quaternion
from loadFile import getPointsCloud # <--- 匯入 getPointsCloud

# --- 常數設定 ---
MODEL_PATH = './data/model.h5'  # 確認模型路徑正確
NUM_POINTS = 2048  # 點雲中的點數，應與訓練時一致

loaded_model = None

def load_keras_model():
    """載入 Keras 模型"""
    global loaded_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案未找到於: {MODEL_PATH}")
    
    loaded_model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'quaternion_loss': quaternion_loss,
            'l2_normalize_quaternion': l2_normalize_quaternion
        },
        safe_mode=False # 對於 .h5 格式，如果包含自訂 Lambda 層，可能需要設為 False
    )
    print("Keras 模型已成功載入。")
    # loaded_model.summary() # 可選：打印模型摘要以供驗證

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式啟動時載入模型"""
    print("應用程式啟動中...")
    load_keras_model()
    yield
    print("應用程式關閉中...")

app = FastAPI(title="Jaw Direction Prediction API", lifespan=lifespan)

class PredictionOutput(BaseModel):
    quaternion: List[float] # 預測的四元數 [w, x, y, z] 或 [x, y, z, w] 取決於您的模型輸出

@app.post("/predict/", response_model=PredictionOutput, summary="上傳STL檔案並預測其對應的四元數")
async def predict_jaw_quaternion_from_stl(file: UploadFile = File(..., description="一個STL格式的3D模型檔案")):
    """
    接收一個STL檔案，將其轉換為點雲，然後預測其方向的四元數。
    內部會自動進行點雲的採樣和正規化。
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入，請稍後再試。")

    if not file.filename.lower().endswith(".stl"):
        raise HTTPException(status_code=400, detail="檔案格式錯誤，僅接受STL檔案。")

    # 創建一個臨時檔案來保存上傳的STL
    # tempfile.NamedTemporaryFile 會在 close() 時自動刪除，但 open3d 可能需要檔案路徑持續有效
    # 因此我們手動創建並刪除
    temp_dir = tempfile.mkdtemp()
    temp_stl_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_stl_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 使用 getPointsCloud 處理STL檔案
        # getPointsCloud 內部會處理點雲採樣和正規化
        points_array = getPointsCloud(temp_stl_path) # 這應該回傳 (NUM_POINTS, 3) 的 NumPy 陣列

        if points_array.shape != (NUM_POINTS, 3):
             raise HTTPException(status_code=500, detail=f"點雲轉換後的形狀不符預期。期望 ({NUM_POINTS}, 3)，得到 {points_array.shape}")

        # 模型期望的輸入形狀為 (batch_size, num_points, 3)
        points_to_predict = np.expand_dims(points_array, axis=0)
        prediction = loaded_model.predict(points_to_predict)
        predicted_quaternion = prediction[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理STL檔案或預測時發生錯誤: {str(e)}")
    finally:
        # 清理臨時檔案和目錄
        if os.path.exists(temp_stl_path):
            os.remove(temp_stl_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        await file.close() # 確保關閉上傳的檔案流

    return PredictionOutput(quaternion=predicted_quaternion)

# --- 運行 FastAPI 應用程式 ---
# 若要運行此應用程式，請在終端機中執行:
# python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# 然後您可以訪問 http://localhost:8000/docs 來查看 API 文件並進行測試。