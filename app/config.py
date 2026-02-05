"""
Configuration for IDS Middleware
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App info
    APP_NAME: str = "IDS Middleware API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Deep Learning-based Intrusion Detection System Middleware"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent. parent
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    
    # Model files
    MODEL_STATE_FILE: str = "final_model.pt"
    INFERENCE_ARTIFACTS_FILE: str = "inference_artifacts.pkl"
    TRANSFORM_META_FILE: str = "transform_meta.pkl"  # Changed from .json to .pkl
    FEATURE_COLS_FILE: str = "feature_cols.json"  # 10 features final
    CONFIG_FILE: str = "inference_config.json"  # Changed from config.json
    LABEL_MAP_FILE: str = "label_map.json"
    PER_CLASS_THRESHOLDS_FILE: str = "per_class_thresholds.json"  # Threshold tuning
    REPORT_FILE: str = "results_summary.json"  # Changed from report.json
    
    # API Settings
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Model Settings
    CONFIDENCE_THRESHOLD: float = 0.5  # Fallback threshold (will use per-class thresholds if available)
    USE_THRESHOLD_TUNING: bool = True  # Use per-class optimal thresholds
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()