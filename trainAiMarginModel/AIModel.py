import os
import numpy as np
import tensorflow as tf # 確保 tensorflow 被匯入
from tensorflow import keras
from tensorflow.keras import layers, Model
from scipy.interpolate import interp1d

# --- 模型定義 ---
def create_margin_prediction_model(num_jaw_points, num_target_margin_points, num_tooth_classes):
    # 口掃點雲輸入
    jaw_input = layers.Input(shape=(num_jaw_points, 3), name='jaw_points_input')
    # 牙齒編號輸入 (獨熱編碼)
    tooth_input = layers.Input(shape=(num_tooth_classes,), name='tooth_number_input')

    # 點雲特徵提取分支 (類似PointNet)
    x = layers.Conv1D(64, 1, activation='relu')(jaw_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(512, 1, activation='relu')(x) # 減少一層以簡化
    x = layers.BatchNormalization()(x)
    global_features = layers.GlobalMaxPooling1D(name='global_point_features')(x) # (None, 512)

    # 牙齒編號特徵提取分支
    t = layers.Dense(64, activation='relu')(tooth_input)
    t = layers.Dense(128, activation='relu', name='tooth_features')(t) # (None, 128)

    # 融合特徵
    merged_features = layers.Concatenate(name='merged_features')([global_features, t]) # (None, 512 + 128)

    # 回歸頭，預測邊緣線點
    m = layers.Dense(512, activation='relu')(merged_features)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(256, activation='relu')(m)
    m = layers.Dropout(0.3)(m)
    # 輸出層：num_target_margin_points * 3 個座標值
    margin_output_flat = layers.Dense(num_target_margin_points * 3, activation='linear', name='margin_output_flat')(m)
    # 將扁平輸出重塑為 (num_target_margin_points, 3)
    margin_output = layers.Reshape((num_target_margin_points, 3), name='margin_output')(margin_output_flat)

    model = Model(inputs=[jaw_input, tooth_input], outputs=margin_output, name='MarginPredictor')
    return model