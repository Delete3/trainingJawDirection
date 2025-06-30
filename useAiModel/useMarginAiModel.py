import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import open3d as o3d

# 為了能正確引用專案中的其他模組，進行路徑設置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainAiMarginModel.loadFile import loadOrderData, NUM_MARGIN_POINTS_TARGET
# from trainAiMarginModel.trainAiMarginModel import TOOTH_TO_INDEX, NUM_TOOTH_CLASSES
ALL_TEETH = [
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38
]
TOOTH_TO_INDEX = {tooth: i for i, tooth in enumerate(ALL_TEETH)}
NUM_TOOTH_CLASSES = len(ALL_TEETH)

# --- 1. 加載已訓練好的模型 ---
print("正在加載模型...")
model_path = './data/marginPredictionModel.h5'
if not os.path.exists(model_path):
    print(f"錯誤：找不到模型檔案於 {model_path}")
    sys.exit(1)
model = keras.models.load_model(model_path)
model.summary()
print("模型加載成功。")

# --- 2. 加載並準備預測數據 ---
print("\n正在加載並準備預測數據...")
folderNameArray = os.listdir("./data/200")
# 選擇一個樣本進行預測，這裡我們使用第一個樣本
folderPath = "./data/200/" + folderNameArray[0]
print(f"使用樣本: {folderPath}")

# loadOrderData 返回的是已經歸一化後的點雲
jawPoint_normalized, _, toothNumber, original_center, max_dist = loadOrderData(folderPath)

# 準備模型的第一個輸入：口掃點雲
# 需要增加一個批次維度 (batch dimension)，從 (4096, 3) -> (1, 4096, 3)
jawPoint_input = np.expand_dims(jawPoint_normalized, axis=0)

# 準備模型的第二個輸入：獨熱編碼的牙齒編號
tooth_index = TOOTH_TO_INDEX.get(toothNumber)
if tooth_index is None:
    print(f"錯誤：牙齒編號 {toothNumber} 不在預定義的映射中。")
    sys.exit(1)
one_hot_tooth_number = tf.keras.utils.to_categorical(tooth_index, num_classes=NUM_TOOTH_CLASSES)
# 同樣需要增加一個批次維度，從 (32,) -> (1, 32)
tooth_input = np.expand_dims(one_hot_tooth_number, axis=0)

print("數據準備完成。")

# --- 3. 使用模型進行預測 ---
print("\n正在進行預測...")
predicted_normalized_margin = model.predict([jawPoint_input, tooth_input])
print("預測完成。")

# --- 4. 反歸一化，將結果轉換回物理尺度 ---
# 移除預測結果的批次維度，從 (1, 128, 3) -> (128, 3)
predicted_normalized_margin = predicted_normalized_margin[0]

# 反向操作：先乘以 max_dist，再加上 center
predicted_margin_physical = (predicted_normalized_margin * max_dist) + original_center

print(f"\n預測的物理尺度邊緣線形狀: {predicted_margin_physical.shape}")
print("預測邊緣線的前5個點座標:")
print(predicted_margin_physical[:5])

# --- 5. 可視化結果 ---
print("\n正在準備可視化...")
# 為了可視化，我們也需要將歸一化的口掃點雲反向轉換回物理尺度
jaw_points_physical_rotated = (jawPoint_normalized * max_dist) + original_center

# 創建口掃點雲對象
jaw_pcd = o3d.geometry.PointCloud()
jaw_pcd.points = o3d.utility.Vector3dVector(jaw_points_physical_rotated)
jaw_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # 灰色

# 創建預測邊緣線的點雲對象
margin_pcd = o3d.geometry.PointCloud()
margin_pcd.points = o3d.utility.Vector3dVector(predicted_margin_physical)
margin_pcd.paint_uniform_color([1, 0, 0]) # 紅色

# 創建線集以連接邊緣線的點，使其更清晰
lines = [[i, i + 1] for i in range(NUM_MARGIN_POINTS_TARGET - 1)]
# lines.append([NUM_MARGIN_POINTS_TARGET - 1, 0]) # 如果想閉合邊緣線，可以取消此行註釋
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(predicted_margin_physical),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.paint_uniform_color([0, 1, 0]) # 綠色

# print("顯示對齊後的口掃模型（灰色）與預測的邊緣線（綠色線條/紅色點）...")
# o3d.visualization.draw_geometries([jaw_pcd, margin_pcd, line_set], window_name="模型預測結果")
