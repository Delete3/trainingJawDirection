import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf # 確保 tensorflow 被匯入
from tensorflow import keras

from loadFile import loadOrderData
# from AIModel import create_pointnet_regression, optimizer, quaternion_loss

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

jawPointsArray = []
marginLineArray = []
toothNumberArray = []

# for i in range(len(folderNameArray)): # 遍歷所有資料夾
for i in range(20): # 遍歷所有資料夾
  folderName = folderNameArray[i]
  folderPath = './data/200/' + folderName
  jawPoint, marginLine, toothNumber = loadOrderData(folderPath)

  jawPointsArray.append(jawPoint)
  marginLineArray.append(marginLine)
  toothNumberArray.append(toothNumber)

jawPointsArray = np.array(jawPointsArray)
marginLineArray = np.array(marginLineArray, dtype=object)
toothNumberArray = np.array(toothNumberArray)
