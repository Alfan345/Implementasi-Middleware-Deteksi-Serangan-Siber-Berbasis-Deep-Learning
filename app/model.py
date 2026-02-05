"""
Model loading and inference (Updated for new model with threshold tuning)
"""
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from app.config import settings


class DNNClassifier(nn.Module):
    """Deep Neural Network for intrusion detection"""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        num_classes: int,
        activation: str = "elu",
        dropout: float = 0.35,
    ):
        super().__init__()

        # Activation functions
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        act_fn = activations.get(activation, nn.ELU())

        # Build layers as Sequential
        layers = []
        prev_size = input_dim

        for hidden_size in layer_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), act_fn, nn.Dropout(dropout)])
            prev_size = hidden_size

        # Output layer (no activation here, logits expected)
        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class IDSModel:
    """Wrapper for IDS model with loading and inference (Updated for threshold tuning)"""

    def __init__(self):
        self.model = None
        self.config = {}
        self.label_map = {}
        self.id_to_label = {}
        self.per_class_thresholds = {}  # Threshold tuning per class
        self.report = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.use_threshold_tuning = settings.USE_THRESHOLD_TUNING

    def load(self, artifacts_dir: Path = None):
        """Load model and all related artifacts"""
        if artifacts_dir is None:
            artifacts_dir = Path(settings.ARTIFACTS_DIR)

        # Try to load inference_artifacts.pkl first (preferred)
        # Note: inference_artifacts.pkl is saved with joblib.dump(), not torch.save()
        inference_artifacts_path = artifacts_dir / settings.INFERENCE_ARTIFACTS_FILE
        if inference_artifacts_path.exists():
            try:
                # Override torch.load to always use map_location for CPU compatibility
                # This fixes CUDA tensors inside joblib-pickled files
                original_torch_load = torch.load
                def patched_torch_load(*args, **kwargs):
                    kwargs.setdefault('map_location', self.device)
                    kwargs.setdefault('weights_only', False)
                    return original_torch_load(*args, **kwargs)
                torch.load = patched_torch_load
                
                try:
                    # Load with joblib (correct method)
                    inference_artifacts = joblib.load(inference_artifacts_path)
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load
            except Exception as e:
                # Fallback to torch.load if joblib fails (for backward compatibility)
                print(f"Warning: joblib.load failed, trying torch.load: {e}")
                inference_artifacts = torch.load(inference_artifacts_path, map_location=self.device, weights_only=False)
            
            self.config = inference_artifacts.get('model_config', {})
            state_dict = inference_artifacts.get('model_state')
            
            # Load label map from inference_artifacts
            if 'label_map' in inference_artifacts:
                self.label_map = inference_artifacts['label_map']
                inv_label_map = inference_artifacts.get('inv_label_map', {})
                # Handle both int keys and string keys
                if inv_label_map:
                    self.id_to_label = {}
                    for k, v in inv_label_map.items():
                        try:
                            self.id_to_label[int(k)] = v
                        except (ValueError, TypeError):
                            self.id_to_label[k] = v
            
            # Load per-class thresholds if available in inference_artifacts
            # (usually stored separately in per_class_thresholds.json)
            if 'per_class_thresholds' in inference_artifacts:
                self.per_class_thresholds = inference_artifacts['per_class_thresholds']
        else:
            # Fallback: load from separate files
            # Load config
            config_path = artifacts_dir / settings.CONFIG_FILE
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, "r") as f:
                config_data = json.load(f)
                self.config = config_data.get('model_config', config_data)
            
            # Load model state
            model_path = artifacts_dir / settings.MODEL_STATE_FILE
            if not model_path.exists():
                raise FileNotFoundError(f"Model state file not found: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)

        # Load label map (if not loaded from inference_artifacts)
        if not self.id_to_label:
            label_map_path = artifacts_dir / settings.LABEL_MAP_FILE
            if label_map_path.exists():
                with open(label_map_path, "r") as f:
                    label_data = json.load(f)
                    self.label_map = label_data.get('label_map', label_data)
                    inv_label_map = label_data.get('inv_label_map', {})
                    # Handle both int keys and string keys
                    self.id_to_label = {}
                    for k, v in inv_label_map.items():
                        try:
                            self.id_to_label[int(k)] = v
                        except (ValueError, TypeError):
                            self.id_to_label[k] = v

        # Load per-class thresholds for threshold tuning (if not already loaded from inference_artifacts)
        if not self.per_class_thresholds:
            thresholds_path = artifacts_dir / settings.PER_CLASS_THRESHOLDS_FILE
            if thresholds_path.exists() and self.use_threshold_tuning:
                try:
                    with open(thresholds_path, 'r') as f:
                        thresholds_data = json.load(f)
                        # Handle both formats: {class: threshold} or {class: {threshold: value}}
                        for class_name, threshold_value in thresholds_data.items():
                            if isinstance(threshold_value, dict):
                                self.per_class_thresholds[class_name] = threshold_value.get('threshold', 0.5)
                            else:
                                self.per_class_thresholds[class_name] = float(threshold_value)
                    print(f"Loaded per-class thresholds: {list(self.per_class_thresholds.keys())}")
                except Exception as e:
                    print(f"Warning: Failed to load per-class thresholds: {e}")
                    self.per_class_thresholds = {}
            else:
                if self.use_threshold_tuning:
                    print("Per-class thresholds not found, using argmax")
        # Load report (optional)
        report_path = artifacts_dir / settings.REPORT_FILE
        if report_path.exists():
            with open(report_path, "r") as f:
                try:
                    self.report = json.load(f)
                except Exception:
                    self.report = {}

        # Build model architecture from config
        input_dim = int(self.config.get("input_dim", 0))
        layer_sizes = list(self.config.get("hidden_layers", self.config.get("layers", [])))
        num_classes = int(self.config.get("num_classes", len(self.id_to_label) if self.id_to_label else 0))

        # Create DNNClassifier and load state dict
        self.model = DNNClassifier(
            input_dim=input_dim,
            layer_sizes=layer_sizes,
            num_classes=num_classes,
            activation=self.config.get("activation", "elu"),
            dropout=self.config.get("dropout", 0.35),
        )

        # Try to load state dict (handle various key formats)
        loaded_ok = False
        def try_strip_prefixes(sd):
            stripped = {}
            for k, v in sd.items():
                new_key = k
                if k.startswith("model."):
                    new_key = k[len("model."):]
                elif k.startswith("network."):
                    new_key = k[len("network."):]
                elif k.startswith("module."):
                    new_key = k[len("module."):]
                stripped[new_key] = v
            return stripped

        # attempt 0: direct load
        try:
            self.model.load_state_dict(state_dict)
            loaded_ok = True
        except Exception:
            pass

        if not loaded_ok:
            # attempt 1: strip prefixes
            new_state = try_strip_prefixes(state_dict)
            try:
                self.model.load_state_dict(new_state)
                loaded_ok = True
            except Exception:
                pass

        if not loaded_ok:
            expected_keys = set(self.model.state_dict().keys())
            actual_keys = set(state_dict.keys())
            raise RuntimeError(
                f"Failed loading state_dict into model.\n"
                f"Sample expected keys: {list(expected_keys)[:5]}\n"
                f"Sample actual keys: {list(actual_keys)[:5]}"
            )

        # Move to device and set eval
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        print(f"Model loaded on {self.device}")
        print(f"   Architecture layers: {layer_sizes}")
        print(f"   Num classes: {num_classes}")
        print(f"   Classes: {list(self.id_to_label.values())}")
        print(f"   Threshold tuning: {'Enabled' if self.per_class_thresholds else 'Disabled (argmax)'}")

    def _apply_threshold_tuning(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply per-class threshold tuning to probabilities
        
        Args:
            probs: numpy array of shape (n_samples, n_classes) with probabilities
            
        Returns:
            preds: numpy array of shape (n_samples,) with predicted class IDs
        """
        if not self.per_class_thresholds or not self.use_threshold_tuning:
            # Fallback to argmax
            return np.argmax(probs, axis=1).astype(int)
        
        n_samples = probs.shape[0]
        preds = np.zeros(n_samples, dtype=int)
        
        # Create label to ID mapping
        label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        # For each sample, find class with probability >= threshold
        for i in range(n_samples):
            best_class_id = None
            best_prob = -1.0
            
            # Check each class
            for class_id in range(probs.shape[1]):
                class_label = self.id_to_label.get(class_id, str(class_id))
                threshold = self.per_class_thresholds.get(class_label, 0.5)
                
                if probs[i, class_id] >= threshold and probs[i, class_id] > best_prob:
                    best_prob = probs[i, class_id]
                    best_class_id = class_id
            
            # If no class meets threshold, use argmax
            if best_class_id is None:
                best_class_id = np.argmax(probs[i])
            
            preds[i] = best_class_id
        
        return preds

    def _tensor_from_numpy(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on preprocessed features
        
        Returns:
            preds: np.ndarray shape (n,) - predicted class IDs
            probs: np.ndarray shape (n, num_classes) - probabilities for each class
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = self._tensor_from_numpy(np.asarray(features))
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Apply threshold tuning
            preds = self._apply_threshold_tuning(probs)

        return preds, probs

    def predict_single(self, features: np.ndarray) -> Dict:
        """
        Predict single flow and return detailed result
        features: 1D numpy array or 2D with single row
        """
        arr = np.asarray(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        preds, probs = self.predict(arr)
        pred_id = int(preds[0])
        pred_label = self.id_to_label.get(pred_id, str(pred_id))
        confidence = float(probs[0, pred_id])

        prob_dict = {self.id_to_label.get(i, str(i)): float(probs[0, i]) for i in range(probs.shape[1])}

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "is_attack": pred_label != "BENIGN",
            "probabilities": prob_dict,
        }

    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """Predict batch of flows (features: 2D numpy array or array-like)"""
        arr = np.asarray(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        preds, probs = self.predict(arr)

        results = []
        for i in range(len(preds)):
            pred_id = int(preds[i])
            pred_label = self.id_to_label.get(pred_id, str(pred_id))
            confidence = float(probs[i, pred_id])
            prob_dict = {self.id_to_label.get(j, str(j)): float(probs[i, j]) for j in range(probs.shape[1])}
            results.append(
                {
                    "prediction": pred_label,
                    "confidence": confidence,
                    "is_attack": pred_label != "BENIGN",
                    "probabilities": prob_dict,
                }
            )

        return results

    def get_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": "DNN-IDS",
            "version": getattr(settings, "APP_VERSION", "unknown"),
            "input_features": int(self.config.get("input_dim", 0)),
            "num_classes": int(self.config.get("num_classes", len(self.id_to_label))),
            "classes": list(self.id_to_label.values()),
            "architecture": {
                "layers": self.config.get("hidden_layers", self.config.get("layers", [])),
                "activation": self.config.get("activation", "elu"),
                "dropout": self.config.get("dropout", 0.35),
            },
            "performance": {
                "macro_f1": self.report.get("macro_f1"),
                "accuracy": self.report.get("accuracy"),
            },
            "threshold_tuning": {
                "enabled": bool(self.per_class_thresholds),
                "thresholds": self.per_class_thresholds,
            },
        }


# Global model instance
ids_model = IDSModel()

