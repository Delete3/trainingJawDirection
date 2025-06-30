from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
from typing import List
import tempfile
import shutil
from contextlib import asynccontextmanager
from scipy.spatial.transform import Rotation as R
from fastapi.middleware.cors import CORSMiddleware

# 為了能正確引用專案中的其他模組，進行路徑設置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 從專案中匯入必要的模組和函數 ---

# 用於牙弓方向預測模型
from trainAiModel.AIModel import quaternion_loss, l2_normalize_quaternion
from trainAiModel.loadFile import getPointsCloud as get_points_for_direction_model

# 用於邊緣線預測模型
from trainAiMarginModel.loadFile import getPoints as get_raw_points_from_stl
from trainAiMarginModel.loadFile import normalizeJawPoint
from trainAiMarginModel.trainAiMarginModel import ALL_TEETH, TOOTH_TO_INDEX, NUM_TOOTH_CLASSES

# --- 全域變數，用於存放已載入的模型 ---
jaw_direction_model = None
margin_prediction_model = None

# --- 在應用程式啟動時載入模型 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """在應用程式啟動時載入模型"""
    global jaw_direction_model, margin_prediction_model
    
    print("正在載入牙弓方向預測模型...")
    jaw_direction_model_path = './data/model.h5'
    if not os.path.exists(jaw_direction_model_path):
        raise FileNotFoundError(f"模型檔案未找到: {jaw_direction_model_path}")
    jaw_direction_model = keras.models.load_model(
        jaw_direction_model_path,
        custom_objects={
            'quaternion_loss': quaternion_loss,
            'l2_normalize_quaternion': l2_normalize_quaternion
        },
        safe_mode=False
    )
    print("牙弓方向預測模型已載入。")

    print("正在載入邊緣線預測模型...")
    margin_model_path = './data/marginPredictionModel.h5'
    if not os.path.exists(margin_model_path):
        raise FileNotFoundError(f"模型檔案未找到: {margin_model_path}")
    margin_prediction_model = keras.models.load_model(margin_model_path)
    print("邊緣線預測模型已載入。")
    
    yield
    print("應用程式關閉中...")

# --- FastAPI 應用程式實例 ---
app = FastAPI(title="牙弓與邊緣線整合預測 API", lifespan=lifespan)

# --- CORS 中間件設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 模型定義 ---
class MarginPredictionOutput(BaseModel):
    margin_line: List[List[float]]

# --- API 端點 ---
@app.post("/predict-margin/", response_model=MarginPredictionOutput, summary="上傳STL檔案和齒位，預測邊緣線")
async def predict_margin(
    tooth_number: int = Form(..., description=f"要預測的牙齒編號，例如: 11。有效範圍: {ALL_TEETH}"),
    file: UploadFile = File(..., description="一個STL格式的3D口掃模型檔案")
):
    """
    接收一個STL口掃模型和牙齒編號，執行以下步驟：
    1. 預測口掃模型的標準方向。
    2. 根據預測的方向對齊模型並進行正規化。
    3. 結合牙齒編號預測出3D邊緣線。
    4. 將邊緣線反正規化後回傳。
    """
    if not jaw_direction_model or not margin_prediction_model:
        raise HTTPException(status_code=503, detail="模型尚未載入，請稍後再試。")

    if not file.filename.lower().endswith(".stl"):
        raise HTTPException(status_code=400, detail="檔案格式錯誤，僅接受STL檔案。")
        
    if tooth_number not in TOOTH_TO_INDEX:
        raise HTTPException(status_code=400, detail=f"無效的牙齒編號: {tooth_number}。有效範圍: {ALL_TEETH}")

    # 創建臨時檔案來處理上傳的STL
    temp_dir = tempfile.mkdtemp()
    temp_stl_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_stl_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- 步驟 1: 預測方向四元數 ---
        # 此函數應對點雲進行方向預測模型所需的正規化
        points_for_direction = get_points_for_direction_model(temp_stl_path)
        points_for_direction_batch = np.expand_dims(points_for_direction, axis=0)
        predicted_quat_array = jaw_direction_model.predict(points_for_direction_batch)
        
        # 將預測結果轉換為 Scipy Rotation 物件
        predicted_rotation = R.from_quat(predicted_quat_array[0])

        # --- 步驟 2: 讀取原始點雲並進行對齊與正規化 ---
        # 此函數僅對點雲進行採樣，不進行正規化
        raw_points = get_raw_points_from_stl(temp_stl_path)
        
        # 使用預測出的四元數對齊點雲，然後進行正規化
        # 我們傳入一個虛擬的 margin line，因為它在正規化牙弓時不會被使用
        normalized_jaw_points, _, center, max_dist = normalizeJawPoint(
            raw_points.copy(), 
            np.zeros((10, 3)), # 虛擬的 margin line
            predicted_rotation
        )
        
        # --- 步驟 3: 準備牙齒編號的獨熱編碼輸入 ---
        tooth_index = TOOTH_TO_INDEX.get(tooth_number)
        one_hot_tooth = tf.keras.utils.to_categorical(tooth_index, num_classes=NUM_TOOTH_CLASSES)

        # --- 步驟 4: 預測邊緣線 ---
        # 為模型輸入增加批次維度
        jaw_input_batch = np.expand_dims(normalized_jaw_points, axis=0)
        tooth_input_batch = np.expand_dims(one_hot_tooth, axis=0)
        
        predicted_normalized_margin = margin_prediction_model.predict([jaw_input_batch, tooth_input_batch])
        
        # --- 步驟 5: 反正規化預測出的邊緣線 ---
        # 移除批次維度並執行反正規化
        predicted_margin_physical = (predicted_normalized_margin[0] * max_dist) + center
        
        return MarginPredictionOutput(margin_line=predicted_margin_physical.tolist())

    except Exception as e:
        # 記錄詳細錯誤以供調試
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"處理過程中發生內部錯誤: {str(e)}")
    finally:
        # 清理臨時檔案
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        await file.close()

# --- 運行 FastAPI 應用程式的指令 ---
# 在終端機中執行:
# uvicorn useAiModel.useAsApi2:app --reload --host 0.0.0.0 --port 8001

