## 安裝新版python

https://www.pcschool.com.tw/blog/it/python-installation

---

## 安裝開發所需套件

`pip install open3d tensorflow scikit-learn scipy fastapi pydantic python-multipart uvicorn`

---

## 獲得訓練資料

### 直接於ftp下載（推薦）
於ftp以下目錄可下載：/資料交換區/200筆AI訓練資料/200  下載后放入放入`./data`資料夾

### 利用nodejs工具撈取
1. 將cmd切換到airdesign-fetchOrder目錄下: `cd airdesign-fetchOrder`
2. 安裝工具包所需套件: `npm install`
3. 執行工具包: `node index.js`，下載完成後會出現data資料夾，底下是200個亂碼的資料夾
4. 回到上層專案根目錄trainingJawDirection，在data資料夾下建立`200`資料夾，並將剛剛下載的200個亂碼的資料夾移入

---

## 訓練偵測口掃方向AI模型

於專案目錄下，執行`python trainAiModel\trainAiModel.py`

## 載入訓練好的AI模型

於專案目錄下，執行`python useAiModel\useAiModel.py`

## 以API模式載入AI模型

於專案目錄下，執行`python -m uvicorn useAiModel.useAsApi:app --reload --host 0.0.0.0 --port 8000`

## 呼叫API預測模型方向
成功執行啓動服務後，用postman呼叫`put http://localhost:8000/predict/`，以formdata傳入key為file的stl、ply、obj模型檔，就可獲得含四個數的矩陣代表四元數
或安裝並啓動以下專案：https://github.com/Delete3/AiModelResultVerifying
