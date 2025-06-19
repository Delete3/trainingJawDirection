import open3d as o3d
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

NUM_MARGIN_POINTS_TARGET = 128  # 重採樣後邊緣線的目標點數
NUM_JAW_POINTS = 4096  # 口掃點雲的點數


# --- 輔助函數：重採樣邊緣線點 ---
def resample_margin_line(points, num_target_points):
    if points is None or len(points) == 0:
        print("警告：接收到空的邊緣線點進行重採樣，將返回零點。")
        return np.zeros((num_target_points, 3))
    if len(points) == 1:  # 如果只有一個點，則複製它
        return np.repeat(points, num_target_points, axis=0)

    # 創建原始點的參數化（例如，沿線的累積索引）
    original_indices = np.arange(len(points))
    # 創建目標點的參數化索引
    target_indices = np.linspace(0, len(points) - 1, num_target_points)

    resampled = np.zeros((num_target_points, 3))
    for i in range(3):  # 對 x, y, z 分別進行插值
        try:
            interpolator = interp1d(
                original_indices, points[:, i], kind="linear", fill_value="extrapolate"
            )
            resampled[:, i] = interpolator(target_indices)
        except ValueError as e:
            print(
                f"插值錯誤 ({['x','y','z'][i]}): {e}. 原始點數: {len(points)}. 點: {points[:5]}"
            )
            # 如果插值失敗（例如，點太少），則使用重複第一個點的方式填充
            resampled[:, i] = points[0, i]
    return resampled


def getPoints(fullpath):
    jawMesh = o3d.io.read_triangle_mesh(fullpath)
    pointCloud = jawMesh.sample_points_uniformly(number_of_points=NUM_JAW_POINTS)
    points = np.asarray(pointCloud.points)
    return points


def normalizeJawPoint(jawPoints, marginPoints, quaternion):
    marginPoints = resample_margin_line(marginPoints, NUM_MARGIN_POINTS_TARGET)

    jawPoints = quaternion.apply(jawPoints)
    marginPoints = quaternion.apply(marginPoints)

    center = np.mean(jawPoints, axis=0)
    jawPoints -= center
    marginPoints -= center

    max_dist = np.max(np.linalg.norm(jawPoints, axis=1))
    jawPoints /= max_dist
    marginPoints /= max_dist

    return jawPoints, marginPoints


def loadQuaternion(fullpath):
    with open(fullpath, "r", encoding="utf-8-sig") as file:
        jsonData = json.loads(file.read())
        matrix = np.array(jsonData)
        rotationMatrix = matrix[:3, :3]
        rotation = R.from_matrix(rotationMatrix)
        quaternion = rotation.as_quat()
        quaternion = R.from_quat(quaternion)
    return quaternion


def loadMarginLine(fullpath):
    with open(fullpath, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()
        toothNumber = int(lines[0].strip().split("_")[1])
        marginPoints = []
        for line in lines[1:-1]:
            parts = line.strip().split()
            marginPoints.append([float(parts[0]), float(parts[1]), float(parts[2])])

    marginPoints = np.array(marginPoints)
    return marginPoints, toothNumber


def loadOrderData(folderPath):
    filePathArray = os.listdir(folderPath)

    for filePath in filePathArray:
        fileNameSplitArray = os.path.basename(filePath).split(".")
        baseName = fileNameSplitArray[0]
        extensionName = fileNameSplitArray[1]
        fullpath = folderPath + "/" + filePath

        if extensionName.lower() == "pts" and "margin_line" in baseName:
            marginLine, toothNumber = loadMarginLine(fullpath)
        elif (extensionName.lower() in ["ply", "stl", "obj"]) and "upper" in baseName:
            upperPoints = getPoints(fullpath)
        elif (extensionName.lower() in ["ply", "stl", "obj"]) and "lower" in baseName:
            lowerPoints = getPoints(fullpath)
        elif extensionName.lower() == "json" and "upper" in baseName:
            upperQuaternion = loadQuaternion(fullpath)
        elif extensionName.lower() == "json" and "lower" in baseName:
            lowerQuaternion = loadQuaternion(fullpath)

    if toothNumber < 30:
        jawPoint = upperPoints
        quaternion = upperQuaternion
    else:
        jawPoint = lowerPoints
        quaternion = lowerQuaternion

    jawPoint, marginLine = normalizeJawPoint(jawPoint, marginLine, quaternion)

    return jawPoint, marginLine, toothNumber
