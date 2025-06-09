import open3d as o3d
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def getPointsCloud(fullpath):
  jawMesh = o3d.io.read_triangle_mesh(fullpath)
  pointCloud = jawMesh.sample_points_uniformly(number_of_points=2048)
  points = np.asarray(pointCloud.points)
  center = np.mean(points, axis=0)
  points -= center
  max_dist = np.max(np.linalg.norm(points, axis=1))
  points /= max_dist
  return points

def loadQuaternion(fullpath):
  with open(fullpath, 'r', encoding='utf-8-sig') as file:
    josn = json.loads(file.read())
    matrix = np.array(josn)
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

folderNameArray = os.listdir('./data/200')
print(folderNameArray)

upperPointsArray = []
lowerPointsArray = []
upperQuaternionArray = []
lowerQuaternionArray = []
for i in range(10):
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