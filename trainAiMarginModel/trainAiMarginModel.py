import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf  # 確保 tensorflow 被匯入
from tensorflow import keras

from loadFile import loadOrderData, NUM_MARGIN_POINTS_TARGET, NUM_JAW_POINTS
from AIModel import create_margin_prediction_model

# --- 固定的牙齒編號映射 ---
# 使用 FDI 牙位表示法 (右上到左上，左下到右下)
ALL_TEETH = [
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38
]
TOOTH_TO_INDEX = {tooth: i for i, tooth in enumerate(ALL_TEETH)}
NUM_TOOTH_CLASSES = len(ALL_TEETH)

print(f"使用固定的牙齒類別，總數: {NUM_TOOTH_CLASSES}")

def getOneHotToothNumberArray(toothNumberArray, tooth_map, num_classes):
    """使用固定的映射將牙齒編號列表轉換為獨熱編碼"""

    oneHotToothNumberArray = []
    for tn in toothNumberArray:
        index = tooth_map.get(tn)
        if index is not None:
            oneHotToothNumberArray.append(tf.keras.utils.to_categorical(index, num_classes=num_classes))
        else:
            print(f"警告：牙齒編號 {tn} 不在預定義的映射中，將被忽略。")
    return np.array(oneHotToothNumberArray)

folderNameArray = os.listdir("./data/200")
jawPointsArray = []
marginLineArray = []
toothNumberArray = []

for i in range(len(folderNameArray)): # 遍歷所有資料夾
    folderName = folderNameArray[i]
    folderPath = "./data/200/" + folderName
    jawPoint, marginLine, toothNumber, _, _ = loadOrderData(folderPath)

    jawPointsArray.append(jawPoint)
    marginLineArray.append(marginLine)
    toothNumberArray.append(toothNumber)

jawPointsArray = np.array(jawPointsArray)
marginLineArray = np.array(marginLineArray)
oneHotToothNumberArray = getOneHotToothNumberArray(toothNumberArray, TOOTH_TO_INDEX, NUM_TOOTH_CLASSES)

print(f"X_jaw_points shape: {jawPointsArray.shape}")
print(f"X_tooth_numbers_one_hot shape: {marginLineArray.shape}")
print(f"Y_margin_lines shape: {oneHotToothNumberArray.shape}")
if jawPointsArray.shape[0] != marginLineArray.shape[0] or jawPointsArray.shape[0] != oneHotToothNumberArray.shape[0]:
    print("錯誤：預處理後的輸入和輸出數據樣本數量不匹配。")
    print(f"Jaw points samples: {jawPointsArray.shape[0]}")
    print(f"Tooth numbers samples: {marginLineArray.shape[0]}")
    print(f"Margin lines samples: {oneHotToothNumberArray.shape[0]}")
    exit()

# --- 劃分訓練集和驗證集 ---
# 我們需要將兩個輸入特徵 (X_jaw_points, X_tooth_numbers_one_hot) 一起劃分
# train_test_split 可以接受列表作為 X
X_jaw_train, X_jaw_val, X_tooth_train, X_tooth_val, Y_margin_train, Y_margin_val = train_test_split(
    jawPointsArray, oneHotToothNumberArray, marginLineArray,
    test_size=0.2, random_state=42
)

print(f"訓練集大小: Jaw={X_jaw_train.shape}, Tooth={X_tooth_train.shape}, Margin={Y_margin_train.shape}")
print(f"驗證集大小: Jaw={X_jaw_val.shape}, Tooth={X_tooth_val.shape}, Margin={Y_margin_val.shape}")

# --- 創建和編譯模型 ---
model = create_margin_prediction_model(NUM_JAW_POINTS, NUM_MARGIN_POINTS_TARGET, NUM_TOOTH_CLASSES)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error', # 或者 'mean_absolute_error'
              metrics=['mae'])

print("開始訓練模型...")

NUM_EPOCHS = 100
BATCH_SIZE = 16 # 可根據您的GPU內存調整
history = model.fit(
    [X_jaw_train, X_tooth_train], Y_margin_train,
    validation_data=([X_jaw_val, X_tooth_val], Y_margin_val),
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    ]
)

val_loss, val_mae = model.evaluate([X_jaw_val, X_tooth_val], Y_margin_val, verbose=0)
print(f"驗證集上的最終損失: {val_loss:.4f}, MAE: {val_mae:.4f}")

model.save('./data/marginPredictionModel.h5')
print("模型已保存為 marginPredictionModel.h5")

# --- 可選：繪製訓練歷史 ---
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='Training MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.title('MAE Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()