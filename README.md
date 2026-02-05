# Implementasi Middleware Deteksi Serangan Siber Berbasis Deep Learning

Middleware berbasis FastAPI untuk deteksi serangan siber menggunakan model Deep Neural Network (DNN) yang dilatih pada dataset CIC-IDS2017.

## 📁 Struktur Proyek

```
├── app/                          # Folder utama aplikasi
│   ├── main.py                   # FastAPI application & endpoints
│   ├── model.py                  # DNN model loading & inference
│   ├── preprocessing.py          # Feature preprocessing pipeline
│   ├── config.py                 # Konfigurasi aplikasi
│   ├── schemas.py                # Pydantic schemas (request/response)
│   ├── column_mapper.py          # Mapper kolom CIC-IDS2017
│   └── dataset_handler.py        # Handler untuk batch prediction
│
├── artifacts/                    # Model & preprocessing artifacts
│   ├── final_model.pt            # Trained DNN model weights
│   ├── transform_meta.pkl        # QuantileTransformer & MinMaxScaler
│   ├── feature_cols.json         # Selected features (10 features)
│   ├── label_map.json            # Label encoding mapping
│   ├── per_class_thresholds.json # Per-class optimal thresholds
│   ├── inference_config.json     # Model configuration
│   └── results_summary.json      # Training results summary
│
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── PANDUAN_MENJALANKAN_SISTEM.md # Panduan lengkap
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/Alfan345/Implementasi-Middleware-Deteksi-Serangan-Siber-Berbasis-Deep-Learning.git
cd Implementasi-Middleware-Deteksi-Serangan-Siber-Berbasis-Deep-Learning

# Buat virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Akses API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/api/v1/model/info

## 📡 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/api/v1/model/info` | Informasi model |
| GET | `/api/v1/features` | Daftar fitur yang diperlukan |
| POST | `/api/v1/predict` | Prediksi single flow |
| POST | `/api/v1/predict/batch` | Prediksi batch flows |
| POST | `/api/v1/predict/dataset` | Prediksi dari file CSV |

## 🔧 Contoh Penggunaan

### Single Flow Prediction

```python
import requests

url = "http://localhost:8000/api/v1/predict"
payload = {
    "features": {
        "Destination Port": 80,
        "Flow Duration": 100000,
        "Total Fwd Packets": 10,
        "Total Length of Fwd Packets": 1500,
        # ... fitur lainnya
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

### Batch Prediction dari CSV

```python
import requests

url = "http://localhost:8000/api/v1/predict/dataset"
files = {"file": open("network_flows.csv", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

## 🏗️ Arsitektur Model

- **Model**: Deep Neural Network (DNN)
- **Input**: 10 fitur network flow yang diseleksi
- **Output**: Klasifikasi multiclass (BENIGN + berbagai jenis serangan)
- **Preprocessing**: QuantileTransformer → MinMaxScaler → Feature Selection

## 📊 Kelas Serangan yang Dideteksi

1. BENIGN (Normal traffic)
2. DDoS
3. PortScan
4. Bot
5. Infiltration
6. Web Attack (Brute Force, XSS, SQL Injection)
7. FTP-Patator
8. SSH-Patator
9. DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)
10. Heartbleed

## 📝 Lisensi

MIT License

## 👤 Author

Alfan - [GitHub](https://github.com/Alfan345)
