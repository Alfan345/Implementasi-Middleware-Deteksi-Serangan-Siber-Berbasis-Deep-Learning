#!/bin/bash
# Script untuk menjalankan middleware dengan conda environment

echo "=========================================="
echo "Starting IDS Middleware Server (Conda)"
echo "=========================================="
echo ""

# Activate conda environment
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate pemodel-env

# Check dependencies
echo "🔍 Checking dependencies..."
if ! python -c "import fastapi, uvicorn, pydantic_settings" 2>/dev/null; then
    echo "⚠️  Installing missing dependencies..."
    pip install fastapi "uvicorn[standard]" python-multipart pydantic pydantic-settings python-dotenv --no-cache-dir
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

# Get external IP if on GCP (with better error handling)
EXTERNAL_IP=$(curl -s -f http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null | head -1)
if [ -z "$EXTERNAL_IP" ] || [[ "$EXTERNAL_IP" == *"html"* ]] || [[ "$EXTERNAL_IP" == *"Error"* ]]; then
    EXTERNAL_IP="localhost"
fi

echo ""
echo "=========================================="
echo "Starting server..."
echo "=========================================="
echo ""
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

