# Panduan Menjalankan Sistem Middleware IDS

## Prasyarat

1. Python 3.10 atau lebih baru
2. CUDA (opsional, untuk GPU acceleration)
3. Artifacts dari preprocessing dan modeling sudah tersedia

---

## Langkah 1: Setup Environment

### A. Buat Virtual Environment (Disarankan)

```bash
cd /home/alfan/test

# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

### B. Install Dependencies

```bash
# Pastikan virtual environment aktif
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies yang akan diinstall:**
- FastAPI (web framework)
- Uvicorn (ASGI server)
- PyTorch (deep learning)
- NumPy, Pandas (data processing)
- Scikit-learn (preprocessing)
- Joblib (model serialization)
- Pydantic (data validation)

---

## Langkah 2: Copy Artifacts

### Opsi A: Menggunakan Script Otomatis (Disarankan)

```bash
cd /home/alfan/test
./copy_artifacts.sh
```

Script akan:
- Copy semua artifacts yang diperlukan
- Membuat `per_class_thresholds.json`
- Verifikasi struktur file

### Opsi B: Manual Copy

```bash
cd /home/alfan/test/artifacts

# Copy dari preprocessing
cp /home/alfan/artifacts_journal_based/transform_meta.pkl .
cp /home/alfan/artifacts_journal_based/feature_cols.json .
cp /home/alfan/artifacts_journal_based/label_map.json .

# Copy dari modeling
cp /home/alfan/artifacts_modeling/final_model.pt .
cp /home/alfan/artifacts_modeling/inference_config.json .
cp /home/alfan/artifacts_modeling/results_summary.json .  # Opsional

# Buat per_class_thresholds.json
cat > per_class_thresholds.json << 'EOF'
{
  "BENIGN": 0.7704,
  "Brute Force": 0.9342,
  "DDoS": 0.9453,
  "DoS": 0.4145,
  "Port Scan": 0.2158
}
EOF
```

### Verifikasi Artifacts

```bash
cd /home/alfan/test/artifacts
ls -lh *.pkl *.json *.pt

# Harus ada:
# - transform_meta.pkl
# - feature_cols.json
# - label_map.json
# - final_model.pt
# - inference_config.json
# - per_class_thresholds.json
```

---

## Langkah 3: Update Middleware (Jika Belum)

### A. Update Preprocessing

Jika `preprocessing.py` masih versi lama, ganti dengan versi baru:

```bash
cd /home/alfan/test/app

# Backup versi lama
cp preprocessing.py preprocessing_old.py

# Ganti dengan versi baru (jika ada preprocessing_new.py)
# Atau update manual sesuai dengan preprocessing_new.py yang sudah dibuat
```

**Catatan**: File `preprocessing_new.py` sudah dibuat sebelumnya. Jika sudah dihapus, perlu dibuat ulang atau update manual.

### B. Update Model Handler

Jika `model.py` masih versi lama, ganti dengan versi baru:

```bash
cd /home/alfan/test/app

# Backup versi lama
cp model.py model_old.py

# Ganti dengan versi baru (jika ada model_new.py)
# Atau update manual sesuai dengan model_new.py yang sudah dibuat
```

**Catatan**: File `model_new.py` sudah dibuat sebelumnya. Jika sudah dihapus, perlu dibuat ulang atau update manual.

### C. Config Sudah Diupdate

File `config.py` sudah diupdate dengan path artifacts yang benar.

---

## Langkah 4: Jalankan Server

### A. Mode Development (Dengan Auto-reload)

```bash
cd /home/alfan/test

# Pastikan virtual environment aktif
source venv/bin/activate  # Linux/Mac

# Jalankan dengan uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Output yang diharapkan:**
```
🚀 Starting IDS Middleware...
✅ Preprocessor loaded:
   - QuantileTransformer: True
   - MinMaxScaler: True
   - Features for transform: 45
   - Features final: 10
✅ Model loaded on cuda
   Architecture layers: [512, 256, 128]
   Num classes: 5
   Classes: ['BENIGN', 'Brute Force', 'DDoS', 'DoS', 'Port Scan']
   Threshold tuning: Enabled
✅ All components loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### B. Mode Production

```bash
cd /home/alfan/test

# Jalankan dengan uvicorn (production)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### C. Menggunakan Python Langsung

```bash
cd /home/alfan/test

# Jalankan langsung dari main.py
python -m app.main
```

---

## Langkah 5: Test Endpoint

### A. Health Check

```bash
curl http://localhost:8000/health
```

**Response yang diharapkan:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### B. Model Info

```bash
curl http://localhost:8000/api/v1/model/info
```

**Response yang diharapkan:**
```json
{
  "model_name": "DNN-IDS",
  "version": "1.0.0",
  "input_features": 10,
  "num_classes": 5,
  "classes": ["BENIGN", "Brute Force", "DDoS", "DoS", "Port Scan"],
  "architecture": {
    "layers": [512, 256, 128],
    "activation": "elu",
    "dropout": 0.35
  },
  "performance": {
    "macro_f1": 0.9207,
    "accuracy": 0.9608
  },
  "threshold_tuning": {
    "enabled": true,
    "thresholds": {
      "BENIGN": 0.7704,
      "Brute Force": 0.9342,
      "DDoS": 0.9453,
      "DoS": 0.4145,
      "Port Scan": 0.2158
    }
  }
}
```

### C. Required Features

```bash
curl http://localhost:8000/api/v1/features
```

**Response yang diharapkan:**
```json
{
  "num_features": 10,
  "features": [
    "max_packet_length",
    "average_packet_size",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "init_win_bytes_backward",
    "packet_length_variance",
    "total_length_of_fwd_packets",
    "bwd_packet_length_max",
    "active_mean",
    "bwd_packet_length_min"
  ]
}
```

### D. Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Destination Port": 80,
      "Flow Duration": 100000,
      "Total Fwd Packets": 10,
      "Total Backward Packets": 5,
      "Total Length of Fwd Packets": 1200,
      "Total Length of Bwd Packets": 600,
      "Fwd Packet Length Max": 150,
      "Fwd Packet Length Min": 100,
      "Bwd Packet Length Max": 120,
      "Bwd Packet Length Min": 80,
      "Fwd Packet Length Mean": 120,
      "Bwd Packet Length Mean": 100,
      "Flow Packets/s": 0.1,
      "Flow Bytes/s": 12,
      "Fwd Packets/s": 0.1,
      "Bwd Packets/s": 0.05,
      "Min Packet Length": 80,
      "Max Packet Length": 150,
      "Packet Length Mean": 110,
      "Packet Length Variance": 500,
      "Packet Length Std": 22.36,
      "Average Packet Size": 110,
      "Fwd Header Length": 40,
      "Bwd Header Length": 40,
      "Fwd Packets/s": 0.1,
      "Bwd Packets/s": 0.05,
      "Min Packet Length": 80,
      "Max Packet Length": 150,
      "Active Mean": 0.5,
      "Active Std": 0.1,
      "Active Max": 1.0,
      "Active Min": 0.0,
      "Idle Mean": 0.2,
      "Idle Std": 0.05,
      "Idle Max": 0.5,
      "Idle Min": 0.0,
      "Fwd IAT Mean": 1000,
      "Fwd IAT Std": 100,
      "Fwd IAT Max": 2000,
      "Fwd IAT Min": 500,
      "Bwd IAT Mean": 2000,
      "Bwd IAT Std": 200,
      "Bwd IAT Max": 4000,
      "Bwd IAT Min": 1000,
      "Fwd Header Length": 40,
      "Bwd Header Length": 40,
      "Init_Win_bytes_forward": 65535,
      "Init_Win_bytes_backward": 65535
    }
  }'
```

**Catatan**: Input harus berisi **45 features** untuk preprocessing, tapi model hanya menggunakan **10 features final**.

### E. Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "flows": [
      {"features": {...}},  # Flow 1
      {"features": {...}}   # Flow 2
    ]
  }'
```

### F. Dataset Upload

```bash
curl -X POST http://localhost:8000/api/v1/predict/dataset \
  -F "file=@dataset_contoh.csv" \
  -F "include_all_results=false"
```

---

## Langkah 6: Akses API Documentation

FastAPI menyediakan dokumentasi interaktif:

1. **Swagger UI**: http://localhost:8000/docs
2. **ReDoc**: http://localhost:8000/redoc

Dokumentasi ini memungkinkan:
- Melihat semua endpoint
- Test endpoint langsung dari browser
- Melihat request/response schema
- Melihat contoh request

---

## Troubleshooting

### Error: "Module not found"

```bash
# Pastikan virtual environment aktif
source venv/bin/activate

# Install ulang dependencies
pip install -r requirements.txt
```

### Error: "Model not loaded" atau "Preprocessor not loaded"

**Penyebab**: Artifacts tidak ditemukan atau path salah

**Solusi**:
```bash
# Check artifacts ada
ls -lh artifacts/*.pkl artifacts/*.json artifacts/*.pt

# Check path di config.py
# Pastikan ARTIFACTS_DIR mengarah ke folder artifacts yang benar
```

### Error: "QuantileTransformer or MinMaxScaler not found"

**Penyebab**: `transform_meta.pkl` tidak berisi transformer

**Solusi**:
```bash
# Verifikasi transform_meta.pkl
python3 << 'EOF'
import joblib
meta = joblib.load('artifacts/transform_meta.pkl')
print("Keys:", list(meta.keys()))
print("Has quantile_transformer:", 'quantile_transformer' in meta)
print("Has minmax_scaler:", 'minmax_scaler' in meta)
EOF

# Jika tidak ada, copy ulang dari artifacts_journal_based
cp /home/alfan/artifacts_journal_based/transform_meta.pkl artifacts/
```

### Error: "Feature X not found in feature_cols_fitted"

**Penyebab**: Input tidak memiliki 45 features yang diperlukan

**Solusi**:
- Pastikan input memiliki semua 45 features (bukan hanya 10)
- Check dengan: `curl http://localhost:8000/api/v1/features`
- Lihat `transform_meta.pkl` untuk daftar 45 features

### Error: Port already in use

```bash
# Gunakan port lain
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Atau kill process yang menggunakan port 8000
lsof -ti:8000 | xargs kill -9  # Linux/Mac
```

### Error: CUDA out of memory

**Solusi**:
- Kurangi batch size di `dataset_handler.py`
- Atau gunakan CPU: set `device = torch.device("cpu")` di `model.py`

---

## Quick Start (Ringkas)

```bash
# 1. Setup
cd /home/alfan/test
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy artifacts
./copy_artifacts.sh

# 3. Jalankan server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Test
curl http://localhost:8000/health

# 5. Akses dokumentasi
# Buka browser: http://localhost:8000/docs
```

---

## Testing dengan Python Script

Buat file `test_api.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Test health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Test model info
response = requests.get(f"{BASE_URL}/api/v1/model/info")
print("Model Info:", json.dumps(response.json(), indent=2))

# Test single prediction
sample_features = {
    "Destination Port": 80,
    "Flow Duration": 100000,
    # ... (tambahkan 45 features)
}

response = requests.post(
    f"{BASE_URL}/api/v1/predict",
    json={"features": sample_features}
)
print("Prediction:", json.dumps(response.json(), indent=2))
```

Jalankan:
```bash
python test_api.py
```

---

## Production Deployment

Untuk production, gunakan:

```bash
# Dengan multiple workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Atau dengan Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Monitoring

### Check Logs

Server akan menampilkan log di console:
- Request/response
- Error messages
- Model loading status

### Health Check Endpoint

Gunakan `/health` untuk monitoring:
```bash
curl http://localhost:8000/health
```

---

## Kesimpulan

1. ✅ Setup environment dan install dependencies
2. ✅ Copy artifacts (6 file wajib)
3. ✅ Jalankan server dengan `uvicorn`
4. ✅ Test endpoint dengan curl atau browser
5. ✅ Akses dokumentasi di `/docs`

Sistem siap digunakan! 🚀

