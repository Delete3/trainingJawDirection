import open3d as o3d
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def getPointsCloud(fullpath):
  jawMesh = o3d.io.read_triangle_mesh(fullpath)
  pointCloud = jawMesh.sample_points_uniformly(number_of_points=4096)
  points = np.asarray(pointCloud.points)
  center = np.mean(points, axis=0)
  points -= center
  max_dist = np.max(np.linalg.norm(points, axis=1))
  points /= max_dist
  return points

def loadQuaternion(fullpath):
  with open(fullpath, 'r', encoding='utf-8-sig') as file:
    jsonData = json.loads(file.read())
    matrix = np.array(jsonData)
    rotationMatrix = matrix[:3,:3]
    rotation = R.from_matrix(rotationMatrix)
    quaternion = rotation.as_quat()
  return quaternion

def loadOrderData(folderPath):
  filePathArray = os.listdir(folderPath)
  for filePath in filePathArray:
    fileNameSplitArray = os.path.basename(filePath).split('.')
    baseName = fileNameSplitArray[0]
    extensionName = fileNameSplitArray[1]
    fullpath = folderPath + '/' + filePath

    if (extensionName.lower() in ['ply', 'stl', 'obj']) and 'upper' in baseName:
      upperPoints = getPointsCloud(fullpath)
    elif (extensionName.lower() in ['ply', 'stl', 'obj']) and 'lower' in baseName:
      lowerPoints = getPointsCloud(fullpath)
    elif extensionName.lower() == 'json' and 'upper' in baseName:
      upperQuaternion = loadQuaternion(fullpath)
    elif extensionName.lower() == 'json' and 'lower' in baseName:
      lowerQuaternion = loadQuaternion(fullpath)

  return upperPoints, upperQuaternion, lowerPoints, lowerQuaternion

# 這部分代碼看起來是用於載入訓練數據，API 服務本身可能不需要在啟動時執行它。
# 如果 API 服務不需要這些預載的數據，可以將其註解或移至訓練腳本中。
# folderNameArray = os.listdir('./data/200')

# upperPointsArray = []
# lowerPointsArray = []
# upperQuaternionArray = []
# lowerQuaternionArray = []
# for i in range(10): # 假設只處理前10個，實際可能需要遍歷所有
#   if i < len(folderNameArray):
#       folderName = folderNameArray[i]
#       folderPath = './data/200/' + folderName
#       upperPoints, upperQuaternion, lowerPoints, lowerQuaternion = loadOrderData(folderPath)
#
#       upperPointsArray.append(upperPoints)
#       lowerPointsArray.append(lowerPoints)
#       upperQuaternionArray.append(upperQuaternion)
#       lowerQuaternionArray.append(lowerQuaternion)
#
# upperPointsArray = np.array(upperPointsArray)
# lowerPointsArray = np.array(lowerPointsArray)
# upperQuaternionArray = np.array(upperQuaternionArray)
# lowerQuaternionArray = np.array(lowerQuaternionArray)