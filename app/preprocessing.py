"""
Preprocessing pipeline for flow features (Updated for new model)
Matches the transform pipeline used during training:
1. QuantileTransformer (output Gaussian)
2. MinMaxScaler (range 0-1)
3. Feature selection (45 -> 10 features)
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

from app.config import settings


class FlowPreprocessor:
    """Preprocessor for network flow features (Updated for new model)"""

    def __init__(self):
        self.quantile_transformer = None
        self.minmax_scaler = None
        self.feature_cols_fitted = None  # 45 features for transformation
        self.feature_cols = None  # 10 features final
        self.medians = None
        self.clip_map = {}  # For extreme value clipping (quantiles)
        self._last_clip_report = {}  # Last clipping operation report
        self.is_loaded = False

    def load(self, artifacts_dir: Path = None):
        """Load preprocessing artifacts"""
        if artifacts_dir is None:
            artifacts_dir = Path(settings.ARTIFACTS_DIR)

        # Load transform_meta.pkl (contains QuantileTransformer, MinMaxScaler, and feature_cols_fitted)
        transform_meta_path = artifacts_dir / settings.TRANSFORM_META_FILE
        if not transform_meta_path.exists():
            raise FileNotFoundError(f"Transform meta file not found: {transform_meta_path}")
        
        transform_meta = joblib.load(transform_meta_path)
        
        self.quantile_transformer = transform_meta.get('quantile_transformer')
        self.minmax_scaler = transform_meta.get('minmax_scaler')
        self.feature_cols_fitted = transform_meta.get('feature_cols_fitted', [])
        
        if self.quantile_transformer is None or self.minmax_scaler is None:
            raise ValueError("QuantileTransformer or MinMaxScaler not found in transform_meta.pkl")
        
        if not self.feature_cols_fitted:
            raise ValueError("feature_cols_fitted not found in transform_meta.pkl")

        # Load feature_cols.json (10 features final)
        feature_cols_path = artifacts_dir / settings.FEATURE_COLS_FILE
        if not feature_cols_path.exists():
            raise FileNotFoundError(f"Feature cols file not found: {feature_cols_path}")
        
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        if not self.feature_cols:
            raise ValueError("feature_cols is empty")

        # Load medians if available (for missing value imputation)
        # Try to get from transform_meta or use default
        self.medians = transform_meta.get('medians', {})
        
        # Load clip_map if available (for extreme value clipping)
        # Format: {feature_name: (q001, q999)} for clipping extreme values
        self.clip_map = transform_meta.get('clip_map', {})
        self.clip_map = transform_meta.get('clip_quantiles', self.clip_map)  # Alternative key name
        
        self.is_loaded = True
        print(f" Preprocessor loaded:")
        print(f"   - QuantileTransformer: {self.quantile_transformer is not None}")
        print(f"   - MinMaxScaler: {self.minmax_scaler is not None}")
        print(f"   - Features for transform: {len(self.feature_cols_fitted)}")
        print(f"   - Features final: {len(self.feature_cols)}")

    def _fill_missing_with_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values using stored training medians"""
        if not self.medians:
            return df
        
        for c in self.feature_cols_fitted:
            if c in df.columns and df[c].isna().any():
                fill_val = self.medians.get(c, df[c].median())
                df[c] = df[c].fillna(fill_val)
        return df

    def transform(self, features: Union[Dict[str, float], pd.DataFrame]) -> np.ndarray:
        """
        Transform input features using the fitted pipeline
        
        Process:
        1. Convert to DataFrame with 45 features (feature_cols_fitted)
        2. Fill missing values with medians
        3. Apply QuantileTransformer
        4. Apply MinMaxScaler
        5. Select 10 features final (feature_cols)
        
        Args:
            features: Dictionary or DataFrame of flow features
            
        Returns:
            Transformed numpy array ready for model inference (10 features)
        """
        if not self.is_loaded:
            raise RuntimeError("Preprocessor not loaded. Call load() first.")

        # Convert to DataFrame if dict
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Ensure all 45 features exist (create NaN for missing)
        for c in self.feature_cols_fitted:
            if c not in df.columns:
                df[c] = np.nan
        
        # Select only 45 features for transformation
        df = df[self.feature_cols_fitted].copy()

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Fill missing values with medians
        if df.isna().any().any():
            df = self._fill_missing_with_medians(df)
            # If still NaN, fill with 0 as fallback
            df = df.fillna(0)

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Convert to numpy array
        X_raw = df.values.astype(np.float32)

        # Step 1: QuantileTransformer (output Gaussian)
        X_qt = self.quantile_transformer.transform(X_raw)

        # Step 2: MinMaxScaler (range 0-1)
        X_scaled = self.minmax_scaler.transform(X_qt).astype(np.float32)

        # Step 3: Feature selection (45 -> 10 features)
        # Get indices of final features in feature_cols_fitted
        feature_indices = []
        for feat in self.feature_cols:
            if feat in self.feature_cols_fitted:
                idx = self.feature_cols_fitted.index(feat)
                feature_indices.append(idx)
            else:
                raise ValueError(f"Feature {feat} not found in feature_cols_fitted")
        
        # Select final 10 features
        X_final = X_scaled[:, feature_indices]

        return X_final

    def transform_batch(self, flows: List[Dict[str, float]]) -> np.ndarray:
        """Transform a batch of flow features"""
        df = pd.DataFrame(flows)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Get list of required feature names (10 features final)"""
        return self.feature_cols.copy()

    def get_feature_names_fitted(self) -> List[str]:
        """Get list of features used for transformation (45 features)"""
        return self.feature_cols_fitted.copy()

    def validate_features(self, features: Dict[str, float]) -> tuple:
        """
        Validate that all required features are present
        
        Note: We need 45 features for transformation, but only 10 are used in final model
        
        Returns:
            (is_valid, missing_features)
        """
        # Check for 45 features needed for transformation
        missing = [col for col in self.feature_cols_fitted if col not in features]
        return len(missing) == 0, missing
    
    def _recompute_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute rate features if base features are available
        This is a placeholder - actual implementation depends on available features
        """
        # If flow_duration is available, recompute rates
        if 'flow_duration' in df.columns:
            flow_duration = df['flow_duration'].replace(0, np.nan)
            if 'total_length_of_fwd_packets' in df.columns and 'total_length_of_bwd_packets' in df.columns:
                total_bytes = df['total_length_of_fwd_packets'] + df['total_length_of_bwd_packets']
                if 'flow_bytes/s' in df.columns:
                    df['flow_bytes/s'] = (total_bytes / flow_duration).fillna(df.get('flow_bytes/s', 0))
            if 'total_fwd_packets' in df.columns and 'total_backward_packets' in df.columns:
                total_packets = df['total_fwd_packets'] + df['total_backward_packets']
                if 'flow_packets/s' in df.columns:
                    df['flow_packets/s'] = (total_packets / flow_duration).fillna(df.get('flow_packets/s', 0))
        return df
    
    def _recompute_packet_length_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute packet length statistics if base features are available
        This is a placeholder - actual implementation depends on available features
        """
        # Recompute max_packet_length if fwd and bwd max are available
        if 'fwd_packet_length_max' in df.columns and 'bwd_packet_length_max' in df.columns:
            if 'max_packet_length' in df.columns:
                df['max_packet_length'] = df[['fwd_packet_length_max', 'bwd_packet_length_max']].max(axis=1)
        
        # Recompute min_packet_length if fwd and bwd min are available
        if 'fwd_packet_length_min' in df.columns and 'bwd_packet_length_min' in df.columns:
            if 'min_packet_length' in df.columns:
                df['min_packet_length'] = df[['fwd_packet_length_min', 'bwd_packet_length_min']].min(axis=1)
        
        # Recompute average_packet_size if mean and std are available
        if 'packet_length_mean' in df.columns:
            if 'average_packet_size' in df.columns:
                df['average_packet_size'] = df['packet_length_mean']
        
        return df
    
    def get_last_clip_report(self) -> Dict:
        """
        Get the last clip report from clipping operations
        Returns empty dict if no clipping was performed
        """
        return getattr(self, '_last_clip_report', {})


# Global preprocessor instance
preprocessor = FlowPreprocessor()

