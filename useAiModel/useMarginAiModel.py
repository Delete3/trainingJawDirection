import numpy as np
import tensorflow as tf
from tensorflow import keras
import os # 為了路徑處理
import json # 為了保存和加載 tooth_to_index

# 從 loadFile.py 導入必要的函數和常數
# 假設 loadFile.py 和 AIModel.py 在同一個目錄或 Python 路徑中
from trainAiMarginModel.loadFile import getPoints, loadQuaternion, normalizeJawPoint, NUM_JAW_POINTS, NUM_MARGIN_POINTS_TARGET

# --- 1. 加載模型 ---
# 確保路徑正確
model_path = './data/marginPredictionModel.h5'
if not os.path.exists(model_path):
    print(f"錯誤：找不到模型文件於 {model_path}")
    exit()
try:
    # 如果模型包含自定義對象（例如自定義層或損失函數），需要提供它們
    # 在您的情況下，create_margin_prediction_model 沒有明顯的自定義對象需要傳遞給 load_model
    # 但如果遇到問題，可以嘗試：
    # model = keras.models.load_model(model_path, custom_objects={'CustomLayer': CustomLayer})
    model = keras.models.load_model(model_path)
    print(f"模型 {model_path} 加載成功。")
    model.summary() # 打印模型結構以確認
except Exception as e:
    print(f"加載模型時發生錯誤: {e}")
    exit()


# --- 輔助函數：準備單個樣本的輸入 ---

# 在訓練結束後，您應該保存 tooth_to_index 和 num_tooth_classes
# 這裡我們假設您已經將它們保存到一個 JSON 文件中
TOOTH_MAPPING_FILE = './data/tooth_mapping.json'

def load_tooth_mapping(filepath=TOOTH_MAPPING_FILE):
    if not os.path.exists(filepath):
        print(f"錯誤: 找不到牙齒映射文件 {filepath}。請確保在訓練後保存了此文件。")
        return None, -1
    with open(filepath, 'r') as f:
        mapping_data = json.load(f)
    return mapping_data['tooth_to_index'], mapping_data['num_tooth_classes']

def prepare_single_input(jaw_points_path, tooth_number_to_predict, quaternion_path, tooth_to_index_map, num_classes):
    """
    準備單個樣本進行預測。
    注意：此函數中的 normalizeJawPoint 調用需要小心處理，
    因為它原本設計用於同時處理 jawPoints 和 marginLine。
    在預測時，我們沒有真實的 marginLine。
    """
    try:
        # a. 加載口掃點雲
        raw_jaw_points = getPoints(jaw_points_path) # 假設 getPoints 返回 (NUM_JAW_POINTS, 3)
        if raw_jaw_points.shape != (NUM_JAW_POINTS, 3):
            print(f"錯誤：加載的口掃點雲形狀不正確 {raw_jaw_points.shape}")
            return None, None, None, None

        # b. 加載四元數
        quaternion = loadQuaternion(quaternion_path) # 返回 scipy Rotation 物件

        # c. 歸一化口掃點雲
        # 對於 normalizeJawPoint，我們需要提供一個 marginPoints 參數。
        # 因為我們正在預測邊緣線，所以我們沒有真實的 marginPoints。
        # 我們可以傳入一個虛擬的、符合預期形狀的零數組，或者修改 normalizeJawPoint。
        # 這裡我們傳入一個虛擬的 marginPoints，因為 normalizeJawPoint 內部會對其重採樣。
        # 重要的是 jawPoints 的歸一化過程。
        # 同時，我們需要獲取歸一化時使用的 center 和 max_dist 以便後續反向轉換。

        # 修改 normalizeJawPoint 以返回 center 和 max_dist
        # 或者創建一個僅用於預測的歸一化版本
        # 為了簡化，我們假設 normalizeJawPoint 已經被修改或我們接受其對虛擬 marginLine 的處理

        # 獲取原始點雲的中心和最大距離，以便後續反向轉換
        # 注意：這裡的 center 和 max_dist 是在 *旋轉後*，但在 *平移和縮放前* 計算的
        # 如果要完全還原，需要在 normalizeJawPoint 內部獲取這些值
        temp_rotated_jaw = quaternion.apply(raw_jaw_points.copy()) # 先旋轉
        original_center = np.mean(temp_rotated_jaw, axis=0)
        temp_centered_jaw = temp_rotated_jaw - original_center
        original_max_dist = np.max(np.linalg.norm(temp_centered_jaw, axis=1))

        # 假設 normalizeJawPoint 接受 None 作為 marginPoints 或我們傳入虛擬值
        # 並且它只返回歸一化的 jaw_points
        # 理想情況下，normalizeJawPoint 應該分離 jaw 和 margin 的處理，或者有一個預測模式
        normalized_jaw_points, _ = normalizeJawPoint(raw_jaw_points.copy(), np.zeros((10,3)), quaternion) # 傳入虛擬 margin

        # d. 處理牙齒編號
        if tooth_number_to_predict not in tooth_to_index_map:
            print(f"錯誤：牙齒編號 {tooth_number_to_predict} 不在已知的映射中。")
            return None, None, None, None
        tooth_index = tooth_to_index_map[str(tooth_number_to_predict)] # JSON key 可能是字串
        one_hot_tooth_number = tf.keras.utils.to_categorical(tooth_index, num_classes=num_classes)

        # e. 調整輸入形狀以匹配模型期望 (batch_size, num_points, 3) 和 (batch_size, num_classes)
        normalized_jaw_points_batch = np.expand_dims(normalized_jaw_points, axis=0)
        one_hot_tooth_number_batch = np.expand_dims(one_hot_tooth_number, axis=0)

        return normalized_jaw_points_batch, one_hot_tooth_number_batch, original_center, original_max_dist

    except Exception as e:
        print(f"準備輸入數據時發生錯誤: {e}")
        return None, None, None, None

def postprocess_prediction(predicted_normalized_margin, center, max_dist, inverse_quaternion=None):
    """
    將預測的歸一化邊緣線轉換回原始尺度。
    如果提供了逆四元數，則也應用逆旋轉。
    """
    # 移除批次維度
    if predicted_normalized_margin.ndim == 3 and predicted_normalized_margin.shape[0] == 1:
        predicted_normalized_margin = predicted_normalized_margin[0]

    # 反向縮放和平移
    predicted_margin_centered = predicted_normalized_margin * max_dist
    predicted_margin_original_scale_rotated = predicted_margin_centered + center

    # 如果需要，應用逆旋轉
    # 注意：center 是在旋轉後的點雲上計算的，所以逆旋轉應該在反向平移和縮放之後應用，
    # 或者更準確地說，center 和 max_dist 應該是針對 *未旋轉* 的原始點雲計算的，
    # 或者，我們將預測結果轉換到與 *已對齊* 的輸入口掃相同的坐標系。
    # 這裡的實現假設我們想得到與 normalizeJawPoint 函數中 jawPoints 最終狀態（歸一化後）
    # 相對應的 margin line，然後再將其轉換回原始物理尺度，但仍在對齊後的姿態。
    # 如果要完全回到最原始的、未對齊的口掃坐標系，還需要應用 quaternion 的逆。

    if inverse_quaternion:
        predicted_margin_original_pose = inverse_quaternion.apply(predicted_margin_original_scale_rotated)
        return predicted_margin_original_pose
    else:
        return predicted_margin_original_scale_rotated


# --- 2. 準備輸入數據 ---
# 示例：假設您有以下新數據的路徑和信息
new_jaw_model_path = "./data/new_sample/upper_jaw.ply" # 替換為您的實際路徑
new_tooth_to_predict = 11 # 要預測的牙齒編號
new_quaternion_path = "./data/new_sample/upper_jaw_orientation.json" # 對齊用的四元數文件路徑

# 加載牙齒編號映射
tooth_to_index_map, num_classes_loaded = load_tooth_mapping()

if tooth_to_index_map is None:
    print("無法加載牙齒映射，預測中止。")
    exit()

# 準備輸入
input_jaw_points, input_one_hot_tooth, pred_center, pred_max_dist = prepare_single_input(
    new_jaw_model_path,
    new_tooth_to_predict,
    new_quaternion_path,
    tooth_to_index_map,
    num_classes_loaded
)

if input_jaw_points is None:
    print("輸入數據準備失敗，預測中止。")
    exit()

print(f"準備好的口掃點雲形狀: {input_jaw_points.shape}")
print(f"準備好的獨熱編碼牙齒編號形狀: {input_one_hot_tooth.shape}")

# --- 3. 進行預測 ---
print("正在進行預測...")
predicted_normalized_margin_line = model.predict([input_jaw_points, input_one_hot_tooth])
# predicted_normalized_margin_line 的形狀將是 (1, NUM_MARGIN_POINTS_TARGET, 3)

print(f"預測的歸一化邊緣線形狀: {predicted_normalized_margin_line.shape}")

# --- 4. 處理預測結果 ---
# 獲取用於反向變換的四元數的逆（如果需要將邊緣線轉回原始未對齊姿態）
# current_quaternion_for_transform = loadQuaternion(new_quaternion_path)
# inverse_rotation = current_quaternion_for_transform.inv()

predicted_margin_line_physical_scale = postprocess_prediction(
    predicted_normalized_margin_line,
    pred_center,
    pred_max_dist
    # inverse_quaternion=inverse_rotation # 如果需要轉回原始姿態，取消此行註釋
)

print(f"預測的物理尺度邊緣線 (前5點):\n{predicted_margin_line_physical_scale[:5]}")

# 現在您可以將 predicted_margin_line_physical_scale 用於可視化或進一步處理
# 例如，使用 open3d 將其與原始（或對齊後的）口掃模型一起顯示

# 可選：可視化 (需要 open3d)
# import open3d as o3d
# jaw_mesh_to_display = o3d.io.read_triangle_mesh(new_jaw_model_path)
# # 如果 jaw_mesh_to_display 未對齊，您可能需要先對其應用 new_quaternion_path 中的旋轉
# # rotated_jaw_mesh = jaw_mesh_to_display.rotate(current_quaternion_for_transform.as_matrix(), center=(0,0,0)) # 假設模型已中心化
#
# margin_pcd = o3d.geometry.PointCloud()
# margin_pcd.points = o3d.utility.Vector3dVector(predicted_margin_line_physical_scale)
# margin_pcd.paint_uniform_color([1, 0, 0]) # 紅色
#
# # 創建線集以連接邊緣線點
# lines = [[i, i + 1] for i in range(len(predicted_margin_line_physical_scale) - 1)]
# # lines.append([len(predicted_margin_line_physical_scale) - 1, 0]) # 可選：閉合邊緣線
# line_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(predicted_margin_line_physical_scale),
#     lines=o3d.utility.Vector2iVector(lines),
# )
# line_set.paint_uniform_color([0, 1, 0]) # 綠色
#
# # o3d.visualization.draw_geometries([jaw_mesh_to_display, margin_pcd, line_set])
# # 如果顯示對齊後的模型：
# # o3d.visualization.draw_geometries([rotated_jaw_mesh, margin_pcd, line_set])


# --- 重要：保存 tooth_mapping.json ---
# 在您的訓練腳本 (trainAiMarginModel.py) 的末尾，您應該添加代碼來保存 tooth_to_index 和 num_tooth_classes
# 這樣預測腳本才能加載它們。
# 例如，在 trainAiMarginModel.py 中：
# ... 訓練完成後 ...
# tooth_mapping_to_save = {
# 'tooth_to_index': {str(k): v for k, v in tooth_to_index.items()}, #確保鍵是字符串以兼容JSON
# 'num_tooth_classes': num_tooth_classes
# }
# with open(TOOTH_MAPPING_FILE, 'w') as f:
# json.dump(tooth_mapping_to_save, f)
# print(f"牙齒映射已保存到 {TOOTH_MAPPING_FILE}")
