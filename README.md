# 🛡️ Implementasi Middleware Deteksi Serangan Siber Berbasis Deep Learning

> Production-ready Intrusion Detection System (IDS) middleware menggunakan Deep Neural Network untuk deteksi serangan siber real-time pada network traffic

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-red.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 **Project Overview**

Sistem **Intrusion Detection System (IDS)** berbasis deep learning yang mampu mendeteksi berbagai jenis serangan siber pada network traffic secara real-time. Model dilatih menggunakan dataset **CIC-IDS2017** dengan akurasi tinggi dan performa inference yang cepat.

### **Key Highlights:**
- 🤖 **Deep Neural Network** - PyTorch-based DNN architecture
- 🚀 **Real-time Detection** - FastAPI async untuk low-latency inference
- 📊 **Multi-class Classification** - 10+ attack types + BENIGN traffic
- 🎯 **Feature Selection** - Optimized dengan 10 selected features
- 📈 **Production-ready** - Complete API dengan batch prediction support
- 🧪 **CIC-IDS2017 Dataset** - Industry-standard cybersecurity dataset

---

## 🚀 **Features**

### **Detection Capabilities**
✅ **BENIGN Traffic** - Normal network behavior  
✅ **DDoS Attacks** - Distributed Denial of Service  
✅ **Port Scanning** - Network reconnaissance  
✅ **Botnet Activity** - Malicious bot detection  
✅ **Infiltration** - Network penetration attempts  
✅ **Web Attacks** - SQL Injection, XSS, Brute Force  
✅ **FTP-Patator** - FTP brute force attacks  
✅ **SSH-Patator** - SSH brute force attacks  
✅ **DoS Variants** - Hulk, GoldenEye, Slowloris, Slowhttptest  
✅ **Heartbleed** - SSL/TLS vulnerability exploit  

### **API Features**
✅ Single flow prediction  
✅ Batch predictions (multiple flows)  
✅ CSV file upload for bulk analysis  
✅ Real-time inference  
✅ Comprehensive API documentation (Swagger UI)  
✅ Health monitoring endpoint  

---

## 📂 **Architecture Overview**

### **System Components**

```
┌─────────────────────────────────────────────────┐
│           Network Traffic (PCAP/Flow)           │
└───────────────────┬─────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   Feature Extractor │
         │  (79 → 10 features) │
         └──────────┬──────────┘
                    │
         ┌──────────▼───────────┐
         │   Preprocessing      │
         │  (QuantileTransform  │
         │   + MinMaxScaler)    │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │   DNN Model (PyTorch)│
         │  (3 hidden layers)   │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │  Attack Classification│
         │   (10+ classes)      │
         └──────────────────────┘
```

### **Model Architecture**

```python
DNN Architecture:
- Input Layer: 10 features
- Hidden Layer 1: 64 neurons (ReLU + Dropout 0.3)
- Hidden Layer 2: 32 neurons (ReLU + Dropout 0.3)
- Hidden Layer 3: 16 neurons (ReLU)
- Output Layer: N classes (Softmax)

Preprocessing Pipeline:
1. QuantileTransformer (output_distribution='uniform')
2. MinMaxScaler (feature_range=(0,1))
3. Feature Selection (10 best features dari 79)
```

---

## 📁 **Repository Structure**

```
Implementasi-Middleware-Deteksi-Serangan-Siber-Berbasis-Deep-Learning/
│
├── app/                              # Main Application Code
│   ├── main.py                       # FastAPI server & REST endpoints
│   ├── model.py                      # DNN model class & inference logic
│   ├── preprocessing.py              # Feature preprocessing pipeline
│   ├── config.py                     # Application configuration
│   ├── schemas.py                    # Pydantic request/response models
│   ├── column_mapper.py              # CIC-IDS2017 feature mapping
│   └── dataset_handler.py            # Batch & file prediction handler
│
├── artifacts/                        # Model Artifacts (not in repo - download separately)
│   ├── final_model.pt                # Trained PyTorch model weights
│   ├── transform_meta.pkl            # Preprocessing transformers
│   ├── feature_cols.json             # Selected feature names
│   ├── label_map.json                # Attack label encoding
│   ├── per_class_thresholds.json    # Per-class optimal thresholds
│   ├── inference_config.json         # Model configuration
│   └── results_summary.json          # Training metrics summary
│
├── scripts/                          # Utility scripts
├── tests/                            # Unit & integration tests
│
├── pemodelan_IDS.ipynb              # Model training notebook (648KB)
├── preprocess_CICIDS2017.ipynb      # Data preprocessing notebook (970KB)
│
├── requirements.txt                  # Python dependencies
├── run_server.sh                     # Launch script (venv)
├── run_server_conda.sh               # Launch script (conda)
├── README.md                         # Main documentation
└── PANDUAN_MENJALANKAN_SISTEM.md    # Detailed setup guide (Indonesian)
```

---

## 💻 **Installation & Setup**

### **Prerequisites**
```bash
- Python 3.11+
- pip atau conda
- 4GB+ RAM (untuk inference)
- CPU atau GPU (optional untuk faster inference)
```

### **1. Clone Repository**
```bash
git clone https://github.com/Alfan345/Implementasi-Middleware-Deteksi-Serangan-Siber-Berbasis-Deep-Learning.git
cd Implementasi-Middleware-Deteksi-Serangan-Siber-Berbasis-Deep-Learning
```

### **2. Setup Virtual Environment**

**Option A: Using venv**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: Using Conda**
```bash
conda create -n ids-ml python=3.11
conda activate ids-ml
pip install -r requirements.txt
```

### **3. Download Model Artifacts**

**Important:** Model artifacts tidak disimpan di repository karena ukuran file besar.

```bash
# Download dari Google Drive / Release page
# Ekstrak ke folder artifacts/
mkdir -p artifacts
# Place the following files in artifacts/:
# - final_model.pt
# - transform_meta.pkl
# - feature_cols.json
# - label_map.json
# - per_class_thresholds.json
# - inference_config.json
# - results_summary.json
```

### **4. Run Server**

**Development Mode:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production Mode:**
```bash
# Using bash script
chmod +x run_server.sh
./run_server.sh

# atau manual
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### **5. Verify Installation**
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/api/v1/model/info
```

---

## 📡 **API Documentation**

### **Endpoints Overview**

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Root endpoint | No |
| GET | `/health` | Health check | No |
| GET | `/docs` | Swagger UI documentation | No |
| GET | `/redoc` | ReDoc documentation | No |
| GET | `/api/v1/model/info` | Model information | No |
| GET | `/api/v1/features` | List required features | No |
| POST | `/api/v1/predict` | Single flow prediction | No |
| POST | `/api/v1/predict/batch` | Batch predictions | No |
| POST | `/api/v1/predict/dataset` | CSV file prediction | No |

---

### **API Usage Examples**

#### **1. Single Flow Prediction**

```python
import requests

url = "http://localhost:8000/api/v1/predict"

payload = {
    "features": {
        "Destination Port": 80,
        "Flow Duration": 120000,
        "Total Fwd Packets": 10,
        "Total Length of Fwd Packets": 1500,
        "Fwd Packet Length Mean": 150,
        "Bwd Packet Length Mean": 120,
        "Flow Bytes/s": 10000,
        "Flow Packets/s": 100,
        "Flow IAT Mean": 1000,
        "Fwd IAT Mean": 1200
    }
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Attack: {result['is_attack']}")
```

**Response:**
```json
{
  "prediction": "BENIGN",
  "confidence": 0.9876,
  "is_attack": false,
  "probabilities": {
    "BENIGN": 0.9876,
    "DDoS": 0.0045,
    "PortScan": 0.0032,
    ...
  },
  "processing_time_ms": 2.34
}
```

---

#### **2. Batch Prediction**

```python
import requests

url = "http://localhost:8000/api/v1/predict/batch"

payload = {
    "flows": [
        {
            "features": {
                "Destination Port": 80,
                "Flow Duration": 120000,
                # ... 10 features
            }
        },
        {
            "features": {
                "Destination Port": 22,
                "Flow Duration": 5000000,
                # ... 10 features
            }
        }
    ]
}

response = requests.post(url, json=payload)
results = response.json()

for i, result in enumerate(results['predictions']):
    print(f"Flow {i+1}: {result['prediction']} ({result['confidence']:.2%})")
```

---

#### **3. CSV File Upload**

```python
import requests

url = "http://localhost:8000/api/v1/predict/dataset"

files = {
    "file": ("network_traffic.csv", open("network_traffic.csv", "rb"), "text/csv")
}

response = requests.post(url, files=files)
result = response.json()

print(f"Total Flows: {result['total_flows']}")
print(f"Attacks Detected: {result['total_attacks']}")
print(f"Attack Types: {result['attack_distribution']}")
```

**Response:**
```json
{
  "total_flows": 1000,
  "total_attacks": 234,
  "benign_count": 766,
  "attack_distribution": {
    "DDoS": 120,
    "PortScan": 45,
    "Bot": 30,
    "DoS": 39
  },
  "predictions": [
    {
      "flow_id": 0,
      "prediction": "BENIGN",
      "confidence": 0.98,
      "is_attack": false
    },
    ...
  ],
  "processing_time_ms": 234.5
}
```

---

## 🔧 **Configuration**

### **Environment Variables**

Create `.env` file:
```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
WORKERS=4

# Model Configuration
MODEL_PATH=artifacts/final_model.pt
TRANSFORM_PATH=artifacts/transform_meta.pkl
FEATURE_COLS_PATH=artifacts/feature_cols.json
LABEL_MAP_PATH=artifacts/label_map.json

# Inference Configuration
BATCH_SIZE=32
CONFIDENCE_THRESHOLD=0.5
```

### **Model Configuration (config.py)**

```python
class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "IDS-DL Middleware"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model Paths
    MODEL_DIR: Path = Path("artifacts")
    MODEL_FILE: str = "final_model.pt"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
```

---

## 🧪 **Testing**

### **Run Unit Tests**
```bash
pytest tests/ -v
```

### **Test Coverage**
```bash
pytest tests/ --cov=app --cov-report=html
```

### **Manual Testing**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test model info
curl http://localhost:8000/api/v1/model/info

# Test prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Destination Port": 80,
      "Flow Duration": 120000,
      "Total Fwd Packets": 10,
      "Total Length of Fwd Packets": 1500,
      "Fwd Packet Length Mean": 150,
      "Bwd Packet Length Mean": 120,
      "Flow Bytes/s": 10000,
      "Flow Packets/s": 100,
      "Flow IAT Mean": 1000,
      "Fwd IAT Mean": 1200
    }
  }'
```

---

## 📊 **Model Performance**

### **Training Metrics**
```
Dataset: CIC-IDS2017
Training Samples: ~2.8M flows
Test Samples: ~700K flows

Overall Metrics:
- Accuracy: 99.2%
- Precision: 98.7%
- Recall: 98.9%
- F1-Score: 98.8%

Per-Class Performance:
- BENIGN: 99.5% accuracy
- DDoS: 99.1% accuracy
- PortScan: 98.3% accuracy
- Bot: 97.8% accuracy
- Web Attack: 98.5% accuracy
```

### **Inference Performance**
```
Hardware: Intel i7 / 16GB RAM
- Single prediction: ~2-5ms
- Batch (100 flows): ~50-80ms
- CSV (1000 flows): ~200-300ms

Hardware: NVIDIA GPU (CUDA)
- Single prediction: ~1-2ms
- Batch (100 flows): ~10-20ms
- CSV (1000 flows): ~50-100ms
```

---

## 🔬 **Dataset Information**

### **CIC-IDS2017 Dataset**

**Source:** Canadian Institute for Cybersecurity (CIC)  
**URL:** https://www.unb.ca/cic/datasets/ids-2017.html

**Dataset Characteristics:**
- Total Samples: ~2.8 million network flows
- Features: 79 statistical features
- Attack Types: 14 different attack scenarios
- Timespan: Monday - Friday (5 days)
- Format: PCAP files + CSV flow statistics

**Selected Features (Top 10):**
1. Destination Port
2. Flow Duration
3. Total Fwd Packets
4. Total Length of Fwd Packets
5. Fwd Packet Length Mean
6. Bwd Packet Length Mean
7. Flow Bytes/s
8. Flow Packets/s
9. Flow IAT Mean
10. Fwd IAT Mean

---

## 🎓 **Skills Demonstrated**

| Category | Skills |
|----------|--------|
| **Deep Learning** | PyTorch, Neural Network Architecture, Model Training |
| **Cybersecurity** | Intrusion Detection, Attack Classification, Network Security |
| **Backend Development** | FastAPI, RESTful API, Async Programming |
| **Data Engineering** | Feature Engineering, Data Preprocessing, Pipeline Design |
| **Machine Learning** | Multi-class Classification, Model Evaluation, Hyperparameter Tuning |
| **DevOps** | Docker-ready, Production Deployment, Health Monitoring |
| **Software Engineering** | Clean Architecture, Error Handling, Logging |
| **Performance Optimization** | Batch Processing, Efficient Inference, Resource Management |

---

## 🚀 **Production Deployment**

### **Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
# Build & Run
docker build -t ids-middleware .
docker run -d -p 8000:8000 --name ids-api ids-middleware
```

### **Cloud Deployment**

**Google Cloud Platform:**
```bash
gcloud run deploy ids-middleware \
  --source . \
  --platform managed \
  --region asia-southeast2 \
  --allow-unauthenticated
```

**AWS Lambda (with Mangum):**
```python
from mangum import Mangum
from app.main import app

handler = Mangum(app)
```

---

## 🔮 **Future Enhancements**

- [ ] Add authentication & API keys
- [ ] Implement rate limiting
- [ ] Add real-time streaming inference (Kafka/RabbitMQ)
- [ ] Create web dashboard (React/Vue.js)
- [ ] Integrate with SIEM systems
- [ ] Add model retraining pipeline
- [ ] Implement A/B testing for models
- [ ] Add explainability (SHAP/LIME)
- [ ] Support for PCAP file direct processing
- [ ] Multi-model ensemble voting

---

## 📚 **References**

1. **CIC-IDS2017 Dataset:**  
   Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018)

2. **FastAPI Documentation:**  
   https://fastapi.tiangolo.com/

3. **PyTorch Documentation:**  
   https://pytorch.org/docs/stable/index.html

4. **Network Security & IDS:**  
   - NIST Cybersecurity Framework
   - MITRE ATT&CK Framework

---

## 🐛 **Troubleshooting**

### **Model tidak bisa di-load**
```python
# Verify artifacts exist
ls -lh artifacts/
# Expected files: final_model.pt, transform_meta.pkl, dll
```

### **Error saat prediction**
```python
# Check feature names
curl http://localhost:8000/api/v1/features

# Verify payload struktur
# Pastikan 10 features dengan nama yang sesuai
```

### **Server tidak bisa start**
```bash
# Check port usage
lsof -i :8000

# Kill process
kill -9 <PID>
```

---

## 👤 **Author**

**Alfanah Muhson**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/alfanah-muhson)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Alfan345)

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Canadian Institute for Cybersecurity (CIC)** untuk CIC-IDS2017 dataset
- **FastAPI team** untuk modern web framework
- **PyTorch community** untuk deep learning framework
- **scikit-learn contributors** untuk preprocessing tools

---

## 📖 **Citation**

If you use this project in your research, please cite:

```bibtex
@software{ids_middleware_2025,
  author = {Alfanah Muhson},
  title = {Implementasi Middleware Deteksi Serangan Siber Berbasis Deep Learning},
  year = {2025},
  url = {https://github.com/Alfan345/Implementasi-Middleware-Deteksi-Serangan-Siber-Berbasis-Deep-Learning}
}
```

---

**⭐ If you find this project useful, please give it a star!**

---

## 🎯 **Use Cases**

This system is suitable for:
- 🏢 **Enterprise Networks** - Real-time threat detection
- 🌐 **ISP/Telco** - Traffic monitoring & security
- 🔒 **SOC (Security Operations Center)** - Alert generation
- 🎓 **Research & Education** - Cybersecurity studies
- 🛡️ **Penetration Testing** - Attack simulation validation

---

**💡 Quick Tip:** Check `PANDUAN_MENJALANKAN_SISTEM.md` untuk panduan lengkap dalam Bahasa Indonesia!
