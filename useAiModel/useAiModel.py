import sys
import os

# 将项目根目录（C:\inteware\trainingJawDirection）添加到 sys.path
# __file__ 是当前脚本的路径: C:\inteware\trainingJawDirection\useAiModel\useAiModel.py
# os.path.dirname(__file__) 是脚本所在的目录: C:\inteware\trainingJawDirection\useAiModel
# os.path.join(os.path.dirname(__file__), '..') 是上一级目录: C:\inteware\trainingJawDirection
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
from tensorflow import keras
# import os # os 已经在上面导入了

import numpy as np # 新增匯入 numpy
# 從 AIModel.py 匯入自訂物件
from trainAiModel.AIModel import quaternion_loss, l2_normalize_quaternion
from trainAiModel.loadFile import loadOrderData

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
