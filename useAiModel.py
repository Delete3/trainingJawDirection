import tensorflow as tf
from tensorflow import keras
import os

import numpy as np # 新增匯入 numpy
# 從 AIModel.py 匯入自訂物件
from AIModel import quaternion_loss, l2_normalize_quaternion
from loadFile import loadOrderData

modelPath = './data/model.h5'
loaded_model = keras.models.load_model(
  modelPath,
  custom_objects={
      'quaternion_loss': quaternion_loss,
      'l2_normalize_quaternion': l2_normalize_quaternion # 提供自訂的正規化函數
  },
  safe_mode=False,
)
loaded_model.summary()

folderNameArray = os.listdir('./data/200')

upperPointsArray = []
# lowerPointsArray = [] # 如果不使用 lowerPointsArray，可以註解或移除
for i in range(5): # 假設您想預測前5個資料夾的數據
  folderName = folderNameArray[i]
  folderPath = './data/200/' + folderName
  # 只需要 upperPoints 來進行預測
  upperPoints, _, _, _ = loadOrderData(folderPath) # 忽略不需要的返回值

  upperPointsArray.append(upperPoints)
  # lowerPointsArray.append(lowerPoints)

# 將 upperPointsArray 轉換為 NumPy 陣列
upperPointsToPredict = np.array(upperPointsArray)

# 進行預測
predictions = loaded_model.predict(upperPointsToPredict)

# 輸出預測結果
print("預測的四元數:")
for i, p in enumerate(predictions):
    print(f"樣本 {i+1}: {p}")
