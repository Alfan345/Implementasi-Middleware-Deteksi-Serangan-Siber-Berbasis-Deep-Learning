#!/bin/bash
# Script untuk menjalankan middleware server

echo "=========================================="
echo "Starting IDS Middleware Server"
echo "=========================================="
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment tidak ditemukan!"
    echo "Membuat virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment dibuat"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "🔍 Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Dependencies belum terinstall!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Check artifacts
echo "🔍 Checking artifacts..."
MISSING_FILES=0

if [ ! -f "artifacts/transform_meta.pkl" ]; then
    echo "  ❌ transform_meta.pkl tidak ditemukan"
    MISSING_FILES=1
fi

if [ ! -f "artifacts/feature_cols.json" ]; then
    echo "  ❌ feature_cols.json tidak ditemukan"
    MISSING_FILES=1
fi

if [ ! -f "artifacts/label_map.json" ]; then
    echo "  ❌ label_map.json tidak ditemukan"
    MISSING_FILES=1
fi

if [ ! -f "artifacts/final_model.pt" ]; then
    echo "  ❌ final_model.pt tidak ditemukan"
    MISSING_FILES=1
fi

if [ ! -f "artifacts/inference_config.json" ]; then
    echo "  ❌ inference_config.json tidak ditemukan"
    MISSING_FILES=1
fi

if [ ! -f "artifacts/per_class_thresholds.json" ]; then
    echo "  ❌ per_class_thresholds.json tidak ditemukan"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "⚠️  Beberapa artifacts tidak ditemukan!"
    echo "Jalankan: ./copy_artifacts.sh"
    echo ""
    read -p "Lanjutkan anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ✅ Semua artifacts ditemukan"
fi

echo ""
echo "=========================================="
echo "Starting server..."
echo "=========================================="
echo ""
# Get external IP if on GCP
EXTERNAL_IP=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null || echo "localhost")

echo "🌐 Server akan berjalan di:"
echo "   - Local: http://localhost:8000"
if [ "$EXTERNAL_IP" != "localhost" ]; then
    echo "   - External: http://$EXTERNAL_IP:8000"
    echo "   ⚠️  Pastikan firewall port 8000 sudah dibuka di GCP!"
fi
echo "📚 Dokumentasi API: http://$EXTERNAL_IP:8000/docs"
echo "🔍 Health check: http://$EXTERNAL_IP:8000/health"
echo ""
echo "Tekan CTRL+C untuk stop server"
echo ""

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

