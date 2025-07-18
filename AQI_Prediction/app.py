# app.py
import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# --- Cấu hình Database ---
# Lấy chuỗi kết nối từ biến môi trường của Render
DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Định nghĩa bảng trong database để lưu kết quả
class AqiRecord(Base):
    __tablename__ = "aqi_records"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, default="esp32-01")
    
     # Dữ liệu đầu vào cho AI model
    pm25_in = Column(Float)
    nh3_in = Column(Float)
    co_in = Column(Float)
    toluene_in = Column(Float)
    
     # Dữ liệu môi trường bổ sung
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    co2 = Column(Float, nullable=True)

    fire_alert = Column(Boolean, default=False) 
     
    # Kết quả dự đoán
    predicted_aqi = Column(Float, nullable=True)
    status = Column(String)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
# Tạo bảng nếu chưa tồn tại
Base.metadata.create_all(bind=engine)

# Hàm để lấy session của database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Tải Models và các hàm chức năng ---
try:
    prediction_model = joblib.load('xgb_model.joblib')
    anomaly_model = joblib.load('anomaly_detector.joblib')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model. Hãy đảm bảo xgb_model.joblib và anomaly_detector.joblib có trong thư mục.")
    # Bạn có thể thêm xử lý lỗi ở đây
    prediction_model = None
    anomaly_model = None
# ... hàm create_new_features giữ nguyên ...
def create_new_features(df_input): # Copy paste hàm của bạn vào đây
    df_output = df_input.copy()
    df_output['PM25_CO_ratio'] = np.divide(df_output['PM2.5'], df_output['CO'] + 1e-6)
    df_output['PM25_NH3_ratio'] = np.divide(df_output['PM2.5'], df_output['NH3'] + 1e-6)
    df_output['Toluene_CO_ratio'] = np.divide(df_output['Toluene'], df_output['CO'] + 1e-6)
    df_output['PM25_x_CO'] = df_output['PM2.5'] * df_output['CO']
    df_output.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_output.fillna(0, inplace=True)
    return df_output

# --- Khởi tạo FastAPI App ---
app = FastAPI(title="AQI Prediction System")

# --- Định nghĩa cấu trúc dữ liệu ---
class SensorData(BaseModel):
     # AI model inputs
    pm25: float
    nh3: float
    co: float
    toluene: float

    fire_alert: bool = False 
    
    # Additional environmental data
    temperature: float | None = None # Dùng | None để cho phép giá trị này có thể không được gửi
    humidity: float | None = None
    co2: float | None = None
    
    device_id: str = "esp32-01"

# --- API Endpoints ---

@app.post("/submit-data")
def submit_sensor_data(data: SensorData, db: Session = Depends(get_db)):
    """
    Endpoint cho ESP32 gửi dữ liệu lên.
    Hàm này sẽ dự đoán và lưu vào DB, không trả về kết quả dự đoán.
    """
    # Quy tắc cứng
    if data.pm25 <= 1 and data.nh3 <= 1 and data.co <= 0.1 and data.toluene <= 0.1:
        status = "Warning: Sensor readings too low, possibly an error."
        aqi_value = None
    elif data.toluene > 100:
        status = "Warning: Toluene level is abnormally high."
        aqi_value = None
    else:
        # Tạo DataFrame và features
        df_for_model = pd.DataFrame({
        'PM2.5': [data.pm25],
        'NH3': [data.nh3],
        'CO': [data.co],
        'Toluene': [data.toluene]
        })
        input_df_featured = create_new_features(df_for_model)

        # Kiểm tra bất thường
        if anomaly_model.predict(input_df_featured)[0] == -1:
            status = "Warning: Input data is anomalous."
            aqi_value = None
        else:
            # Dự đoán
            predicted_aqi_log = prediction_model.predict(input_df_featured)
            aqi_value = round(float(np.expm1(predicted_aqi_log)[0]), 2)
            status = "Success"
    
    # Lưu record vào database
    db_record = AqiRecord(
        device_id=data.device_id,
        # Lưu lại dữ liệu đầu vào để sau này phân tích
        pm25_in=data.pm25,
        nh3_in=data.nh3,
        co_in=data.co,
        toluene_in=data.toluene,
        # Lưu dữ liệu môi trường bổ sung
        temperature=data.temperature,
        humidity=data.humidity,
        co2=data.co2,

        fire_alert=data.fire_alert,
        
        # Lưu kết quả
        predicted_aqi=float(aqi_value) if aqi_value is not None else None,
        status=status
    )
    db.add(db_record)
    db.commit()
    
    return {"message": "Data received successfully."}
    
def get_aqi_category(aqi_value):
    """
    Phân loại giá trị AQI số thành các mức độ ô nhiễm.
    """
    if aqi_value is None:
        return "Không xác định"
    
    aqi = int(aqi_value) # Chuyển thành số nguyên để so sánh

    if 0 <= aqi <= 50:
        return "Tốt"
    elif 51 <= aqi <= 100:
        return "Trung bình"
    elif 101 <= aqi <= 150:
        return "Kém"
    elif 151 <= aqi <= 200:
        return "Xấu"
    elif 201 <= aqi <= 300:
        return "Rất xấu"
    elif aqi > 300:
        return "Nguy hại"
    else:
        return "Không xác định"
        
@app.get("/get-latest-aqi/{device_id}")
def get_latest_aqi_data(device_id: str, db: Session = Depends(get_db)):
    """
    Endpoint cho App Mobile lấy kết quả mới nhất.
    """
    latest_record = db.query(AqiRecord).filter(AqiRecord.device_id == device_id).order_by(AqiRecord.timestamp.desc()).first()
    
    if not latest_record:
        return {"error": "No data found for this device."}

    # Lấy giá trị AQI và tính toán category
    aqi_value = latest_record.predicted_aqi
    aqi_category = get_aqi_category(aqi_value)    

    # Xử lý trường hợp có cảnh báo
    if latest_record.status.startswith("Warning"):
        aqi_category = latest_record.status # Trả về chính cảnh báo đó
        aqi_value = None # Không hiển thị giá trị số nếu có cảnh báo
        
    # Xây dựng cấu trúc JSON trả về
    response_data = {
        "timestamp": latest_record.timestamp.isoformat() + "Z", # Thêm Z để chỉ múi giờ UTC
        
        "fire_alert": latest_record.fire_alert,
        
        "data": {
            "pm25": latest_record.pm25_in,
            "nh3": latest_record.nh3_in,
            "co": latest_record.co_in,
            "toluene": latest_record.toluene_in,
            "temperature": latest_record.temperature,
            "humidity": latest_record.humidity,
            "co2": latest_record.co2,
        },
        "aqi": {
            "value": aqi_value,
            "category": aqi_category
        }
    }
    return response_data
