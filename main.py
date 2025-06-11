from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from typing import List

# 從 AIModel.py 匯入自訂物件
# 確保 AIModel.py 與 main.py 在同一目錄，或 AIModel 在 PYTHONPATH 中
from AIModel import quaternion_loss, l2_normalize_quaternion

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

app = FastAPI(title="Jaw Direction Prediction API")

# --- Pydantic 模型定義 ---
class PointCloudInput(BaseModel):
    # 輸入應為一個點雲，包含 NUM_POINTS 個點，每個點有3個座標 (x, y, z)
    # 例如: [[x1,y1,z1], [x2,y2,z2], ..., [x2048,y2048,z2048]]
    points: List[conlist(float, min_length=3, max_length=3)]

    class Config:
        schema_extra = {
            "example": {
                "points": [[0.1, 0.2, 0.3]] * NUM_POINTS # 範例數據，實際應為真實點雲
            }
        }

class PredictionOutput(BaseModel):
    quaternion: List[float] # 預測的四元數 [w, x, y, z] 或 [x, y, z, w] 取決於您的模型輸出

# --- FastAPI 事件處理 ---
@app.on_event("startup")
async def startup_event():
    """應用程式啟動時載入模型"""
    load_keras_model()

@app.post("/predict/", response_model=PredictionOutput, summary="預測點雲的四元數")
async def predict_jaw_quaternion(data: PointCloudInput):
    """
    接收點雲數據並回傳預測的四元數。
    點雲數據應為一個包含 ${NUM_POINTS} 個點的列表，每個點是一個包含 [x, y, z] 座標的列表。
    **重要提示**: 此 API 端點期望輸入的點雲數據已經過與訓練時相同的正規化處理
    （例如，中心化並按最大距離縮放）。
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入，請稍後再試。")

    if len(data.points) != NUM_POINTS:
        raise HTTPException(status_code=400, detail=f"輸入點雲必須包含 {NUM_POINTS} 個點，但收到了 {len(data.points)} 個點。")

    points_array = np.array(data.points, dtype=np.float32)
    # 模型期望的輸入形狀為 (batch_size, num_points, 3)
    points_to_predict = np.expand_dims(points_array, axis=0)

    prediction = loaded_model.predict(points_to_predict)
    predicted_quaternion = prediction[0].tolist() # 假設模型輸出一個四元數

    return PredictionOutput(quaternion=predicted_quaternion)

# --- 運行 FastAPI 應用程式 ---
# 若要運行此應用程式，請在終端機中執行:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000(試了不行)
# python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000(成功)
#
# 然後您可以訪問 http://localhost:8000/docs 來查看 API 文件並進行測試。