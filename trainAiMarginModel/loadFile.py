import open3d as o3d
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def getPoints(fullpath):
  jawMesh = o3d.io.read_triangle_mesh(fullpath)
  pointCloud = jawMesh.sample_points_uniformly(number_of_points=4096)
  points = np.asarray(pointCloud.points)
  return points

def normalizeJawPoint(jawPoints, marginPoints, quaternion):
  jawPoints = quaternion.apply(jawPoints)
  marginPoints = quaternion.apply(marginPoints)

  center = np.mean(jawPoints, axis=0)
  jawPoints -= center
  marginPoints -= center

  max_dist = np.max(np.linalg.norm(jawPoints, axis=1))
  jawPoints /= max_dist
  marginPoints /= max_dist

  return jawPoints, marginPoints

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
    quaternion = R.from_quat(quaternion)
  return quaternion

def loadMarginLine(fullpath):
  with open(fullpath, 'r', encoding='utf-8-sig') as file:
    lines=file.readlines()
    toothNumber = int(lines[0].strip().split('_')[1])
    marginPoints = []
    for line in lines[1:-1]:
      parts = line.strip().split()
      marginPoints.append([float(parts[0]), float(parts[1]), float(parts[2])])
  
  marginPoints = np.array(marginPoints)
  return marginPoints, toothNumber
    

def loadOrderData(folderPath):
  filePathArray = os.listdir(folderPath)
  
  for filePath in filePathArray:
    fileNameSplitArray = os.path.basename(filePath).split('.')
    baseName = fileNameSplitArray[0]
    extensionName = fileNameSplitArray[1]
    fullpath = folderPath + '/' + filePath

    if extensionName.lower() == 'pts'and 'margin_line'in baseName:
      marginLine, toothNumber = loadMarginLine(fullpath)
    elif (extensionName.lower() in ['ply', 'stl', 'obj']) and 'upper' in baseName:
      upperPoints = getPoints(fullpath)
    elif (extensionName.lower() in ['ply', 'stl', 'obj']) and 'lower' in baseName:
      lowerPoints = getPoints(fullpath)
    elif extensionName.lower() == 'json' and 'upper' in baseName:
      upperQuaternion = loadQuaternion(fullpath)
    elif extensionName.lower() == 'json' and 'lower' in baseName:
      lowerQuaternion = loadQuaternion(fullpath)

  if toothNumber<30:
    jawPoint = upperPoints
    quaternion = upperQuaternion
  else:
    jawPoint = lowerPoints
    quaternion = lowerQuaternion

  jawPoint, marginLine = normalizeJawPoint(jawPoint, marginLine, quaternion)

  return jawPoint, marginLine, toothNumber