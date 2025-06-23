# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="AQI Prediction API")

# --- Tải các model đã được huấn luyện khi khởi động ---
try:
    prediction_model = joblib.load('xgb_model.joblib')
    anomaly_model = joblib.load('anomaly_detector.joblib')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model. Hãy đảm bảo xgb_model.joblib và anomaly_detector.joblib có trong thư mục.")
    # Bạn có thể thêm xử lý lỗi ở đây
    prediction_model = None
    anomaly_model = None

# --- Các hàm chức năng (giống như trong notebook) ---
def create_new_features(df_input):
    df_output = df_input.copy()
    df_output['PM25_CO_ratio'] = np.divide(df_output['PM2.5'], df_output['CO'] + 1e-6)
    df_output['PM25_NH3_ratio'] = np.divide(df_output['PM2.5'], df_output['NH3'] + 1e-6)
    df_output['Toluene_CO_ratio'] = np.divide(df_output['Toluene'], df_output['CO'] + 1e-6)
    df_output['PM25_x_CO'] = df_output['PM2.5'] * df_output['CO']
    df_output.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_output.fillna(0, inplace=True)
    return df_output

# --- Định nghĩa cấu trúc dữ liệu đầu vào cho API ---
class SensorData(BaseModel):
    pm25: float
    nh3: float
    co: float
    toluene: float

# --- Tạo API Endpoint ---
@app.post("/predict")
def predict_aqi_endpoint(data: SensorData):
    """
    Nhận dữ liệu từ cảm biến và trả về dự đoán AQI.
    """
    # Quy tắc cứng
    if data.pm25 <= 1 and data.nh3 <= 1 and data.co <= 0.1 and data.toluene <= 0.1:
        return {"predicted_aqi": None, "status": "Warning: Sensor readings too low, possibly an error."}
    if data.toluene > 100:
        return {"predicted_aqi": None, "status": "Warning: Toluene level is abnormally high."}

    # Tạo DataFrame từ dữ liệu đầu vào
    input_df_original = pd.DataFrame([data.dict()])
    input_df_featured = create_new_features(input_df_original)

    # Kiểm tra bất thường
    is_anomaly = anomaly_model.predict(input_df_featured)
    if is_anomaly[0] == -1:
        return {"predicted_aqi": None, "status": "Warning: Input data is anomalous, prediction may be unreliable."}
    
    # Dự đoán
    predicted_aqi_log = prediction_model.predict(input_df_featured)
    predicted_aqi = np.expm1(predicted_aqi_log)[0]

    return {
        "predicted_aqi": round(predicted_aqi, 2),
        "status": "Success"
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the AQI Prediction API. Use the /predict endpoint for predictions."}