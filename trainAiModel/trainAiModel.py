import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf # 確保 tensorflow 被匯入
from tensorflow import keras

from loadFile import loadOrderData
from AIModel import create_pointnet_regression, optimizer, quaternion_loss

# --- 檢查並設定 GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # 針對每個 GPU 設定內存增長
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"檢測到 {len(gpus)} 個物理 GPU，設定了 {len(logical_gpus)} 個邏輯 GPU")
  except RuntimeError as e:
    # 內存增長必須在 GPU 初始化之前設定
    print(f"設定 GPU 內存增長時發生錯誤: {e}")
else:
    print("未檢測到 GPU，將使用 CPU 進行訓練。")

folderNameArray = os.listdir('./data/200')

upperPointsArray = []
lowerPointsArray = []
upperQuaternionArray = []
lowerQuaternionArray = []
for i in range(len(folderNameArray)): # 遍歷所有資料夾
  folderName = folderNameArray[i]
  folderPath = './data/200/' + folderName
  upperPoints, upperQuaternion, lowerPoints, lowerQuaternion = loadOrderData(folderPath)

  upperPointsArray.append(upperPoints)
  lowerPointsArray.append(lowerPoints)
  upperQuaternionArray.append(upperQuaternion)
  lowerQuaternionArray.append(lowerQuaternion)

upperPointsArray = np.array(upperPointsArray)
lowerPointsArray = np.array(lowerPointsArray)
upperQuaternionArray = np.array(upperQuaternionArray)
lowerQuaternionArray = np.array(lowerQuaternionArray)

upperPointsArrayTrain, upperPointsArrayTemp, upperQuaternionArrayTrain, upperQuaternionArrayTemp = train_test_split(upperPointsArray, upperQuaternionArray, test_size=0.3, random_state=42)
upperPointsArrayVal, upperPointsArrayTest, upperQuaternionArrayVal, upperQuaternionArrayTest = train_test_split(upperPointsArrayTemp, upperQuaternionArrayTemp, test_size=0.5, random_state=42)

print(f"訓練集大小: X={upperPointsArrayTrain.shape}, y={upperQuaternionArrayTrain.shape}")
print(f"驗證集大小: X={upperPointsArrayVal.shape}, y={upperQuaternionArrayVal.shape}")
print(f"測試集大小: X={upperPointsArrayTest.shape}, y={upperQuaternionArrayTest.shape}")

num_points = upperPointsArrayTrain.shape[1]
num_output_values = upperQuaternionArray.shape[1]
model = create_pointnet_regression(num_points, num_output_values)
model.summary()
model.compile(
    optimizer=optimizer,
    loss=quaternion_loss # 使用自定義的四元數損失
)

epochs = 50 # 根據你的數據集大小和複雜度調整
batch_size = 32 # 根據你的 Colab GPU 內存調整

# 確保 y_train 和 y_val 的 shape 與損失函數的期望一致
# 如果使用 geodesic_loss，y_train, y_val 應為 (batch, 9)
history = model.fit(
    upperPointsArrayTrain,
    upperQuaternionArrayTrain, # 真實標籤是展平的 3x3 矩陣
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(upperPointsArrayVal, upperQuaternionArrayVal),
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    ]
)

model.save('./data/model.h5')
model.save('./data/model.keras')