"""
Dataset handler for file upload and batch prediction
(Updated: integrate per-class probability calibration and per-class thresholds while preserving multiclass output)
"""
import io
import time
import re
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException

from app.preprocessing import preprocessor
from app.model import ids_model
from app.column_mapper import SimpleColumnMapper
from app.config import settings

# Confidence threshold for treating non-BENIGN predictions as attacks (legacy fallback).
ATTACK_CONF_THRESHOLD = 0.75

# Auto-fix configuration
AUTO_FIX_EXTREME = True
AUTO_FIX_TOTAL_ROW_FRAC_REJECT = 0.05
AUTO_FIX_PER_FEATURE_FRAC_REJECT = 0.20
EXTREME_FACTOR = 1000.0

# Features allowed to be auto-fixed without triggering a per-feature rejection.
ALLOW_AUTO_FIX = {
    "Fwd IAT Std",
    "Bwd IAT Std",
    "Bwd IAT Mean",
    "Packet Length Variance"
}

# Calibration / thresholds artifacts (filenames inside artifacts dir)
PER_CLASS_THRESH_FILE = "per_class_thresholds.json"
PROB_CALIBRATORS_FILE = "prob_calibrators.pkl"


def _smart_fix_commas(df: pd.DataFrame, train_medians: dict = None, sample_n: int = 100):
    report = {}
    cols = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.contains(',').any()]
    for c in cols:
        s = df[c].astype(str).fillna('')
        sample = s[s != ''].head(sample_n).tolist()
        if not sample:
            continue
        tmpA = s.str.replace(',', '', regex=False).replace({'': None})
        numA = pd.to_numeric(tmpA, errors='coerce')
        tmpB = s.str.replace(',', '.', regex=False).replace({'': None})
        numB = pd.to_numeric(tmpB, errors='coerce')
        naA = int(numA.isna().sum()); naB = int(numB.isna().sum())
        chosen = None; reason = None
        if train_medians and c in train_medians and train_medians[c] is not None:
            train_med = float(train_medians[c])
            def rel_diff(x):
                if x is None or (isinstance(x, float) and np.isnan(x)): return np.inf
                if train_med == 0: return abs(float(x) - train_med)
                return abs((float(x) - train_med) / (train_med + 1e-12))
            medA = numA.median(skipna=True); medB = numB.median(skipna=True)
            dA = rel_diff(medA); dB = rel_diff(medB)
            if dA < dB:
                chosen = numA; reason = f"closer to train median (dA={dA:.3g} < dB={dB:.3g})"
            else:
                chosen = numB; reason = f"closer to train median (dB={dB:.3g} <= dA={dA:.3g})"
        else:
            if naA < naB:
                chosen = numA; reason = f"fewer NaNs ({naA} < {naB})"
            elif naB < naA:
                chosen = numB; reason = f"fewer NaNs ({naB} < {naA})"
            else:
                medA = numA.median(skipna=True) if naA < len(numA) else np.nan
                medB = numB.median(skipna=True) if naB < len(numB) else np.nan
                if abs(medA if not np.isnan(medA) else np.inf) <= abs(medB if not np.isnan(medB) else np.inf):
                    chosen = numA; reason = "tie -> smaller median magnitude (A)"
                else:
                    chosen = numB; reason = "tie -> smaller median magnitude (B)"
        if chosen is None:
            chosen = numA; reason = "fallback remove commas"
        df[c] = chosen
        report[c] = {"na_after": int(chosen.isna().sum()), "median_after": None if chosen.dropna().empty else float(chosen.median()), "choice_reason": reason}
    return df, report


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None


class DatasetHandler:
    SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx']
    MAX_FILE_SIZE_MB = 200

    def __init__(self):
        self.last_dataset: Optional[pd.DataFrame] = None
        self.last_results: Optional[List[Dict]] = None
        self.last_mapping: Optional[Dict] = None
        self.last_summary: Optional[Dict] = None

        # calibration artifacts (lazy loaded)
        self.per_class_thresholds: Dict[str, Dict] = {}
        self.prob_calibrators = None  # joblib dict {'classes': [...], 'calibrators': {class:IsotonicRegression}}
        self._load_calibration_artifacts()

    def _load_calibration_artifacts(self):
        artifacts_dir = Path(settings.ARTIFACTS_DIR)
        # load per-class thresholds if present
        tpath = artifacts_dir / PER_CLASS_THRESH_FILE
        if tpath.exists():
            try:
                with tpath.open('r') as f:
                    self.per_class_thresholds = json.load(f)
                print(f"Loaded per-class thresholds from {tpath}")
            except Exception as e:
                print("Warning: failed to load per-class thresholds:", e)
                self.per_class_thresholds = {}

        # load calibrators if present
        cpath = artifacts_dir / PROB_CALIBRATORS_FILE
        if cpath.exists():
            try:
                self.prob_calibrators = joblib.load(str(cpath))
                print(f"Loaded probability calibrators from {cpath}")
            except Exception as e:
                print("Warning: failed to load prob_calibrators:", e)
                self.prob_calibrators = None

    def _apply_calibration_and_renormalize(self, prob_dicts: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        prob_dicts: list of per-sample dicts {class:prob}
        Apply per-class calibrator (if available), then renormalize per-row so sums to 1.
        Returns list of calibrated dicts in same order.
        """
        if not prob_dicts:
            return prob_dicts
        # collect classes union
        classes = set()
        for d in prob_dicts:
            classes.update(d.keys())
        classes = sorted(classes)
        # build DataFrame
        prob_df = pd.DataFrame([{c: d.get(c, 0.0) for c in classes} for d in prob_dicts])
        # apply calibrators if available
        if self.prob_calibrators and isinstance(self.prob_calibrators, dict):
            cal_classes = self.prob_calibrators.get("classes", [])
            cal_map = self.prob_calibrators.get("calibrators", {})
            for c in classes:
                if c in cal_map:
                    try:
                        # ensure numpy array
                        vals = prob_df[c].astype(float).values
                        transformed = cal_map[c].transform(vals)
                        prob_df[c] = transformed
                    except Exception as e:
                        print(f"Warning: calibrator transform failed for class {c}: {e}")
                        # leave original values
                        continue
        # clip negative/NaN and renormalize per-row
        prob_df = prob_df.fillna(0.0).clip(lower=0.0)
        sums = prob_df.sum(axis=1).replace(0.0, 1.0)
        prob_df = prob_df.div(sums, axis=0)
        # convert back to list of dicts
        calibrated = [ {c: float(prob_df.iat[i, j]) for j,c in enumerate(prob_df.columns)} for i in range(len(prob_df)) ]
        return calibrated

    async def read_file(self, file: UploadFile) -> pd.DataFrame:
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in self.SUPPORTED_FORMATS):
            raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported: {self.SUPPORTED_FORMATS}")
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum: {self.MAX_FILE_SIZE_MB}MB")

        def try_parse_csv(buf):
            candidates = [',',';','\t']
            best_df = None; best_score = -1
            for sep in candidates:
                try:
                    df_try = pd.read_csv(io.BytesIO(buf), sep=sep, thousands=',', quotechar='"', engine='python')
                    n_numeric = 0
                    for c in df_try.columns:
                        coerced = pd.to_numeric(df_try[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
                        n_numeric += int(coerced.notna().sum())
                    if n_numeric > best_score:
                        best_score = n_numeric; best_df = df_try
                except Exception:
                    continue
            return best_df

        try:
            if filename.endswith('.csv'):
                df = try_parse_csv(content)
                if df is None:
                    df = pd.read_csv(io.BytesIO(content), engine='python')
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(content))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail="Unknown file format")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

        df.columns = [c.replace('\ufeff','').strip() for c in df.columns]
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        if not preprocessor.is_loaded:
            try:
                preprocessor.load()
            except Exception:
                pass

        df, comma_report = _smart_fix_commas(df, train_medians=preprocessor.medians)
        if comma_report:
            print("Comma normalization report:", comma_report)

        return df

    def _generate_conclusion(self, summary: Dict) -> Dict:
        total = summary['total_samples']; attack_pct = summary['attack_percentage']; benign_pct = summary['benign_percentage']; attack_breakdown = summary['attack_breakdown']
        if attack_pct == 0:
            threat_level = "AMAN"; threat_color = "green"; threat_description = "Tidak ditemukan aktivitas mencurigakan"
        elif attack_pct < 5:
            threat_level = "RENDAH"; threat_color = "yellow"; threat_description = "Ditemukan sedikit aktivitas mencurigakan"
        elif attack_pct < 20:
            threat_level = "SEDANG"; threat_color = "orange"; threat_description = "Ditemukan aktivitas mencurigakan yang perlu diperhatikan"
        elif attack_pct < 50:
            threat_level = "TINGGI"; threat_color = "red"; threat_description = "Ditemukan banyak aktivitas mencurigakan"
        else:
            threat_level = "KRITIS"; threat_color = "darkred"; threat_description = "Mayoritas traffic terdeteksi sebagai serangan"
        dominant_attack = None; dominant_attack_count = 0; dominant_attack_pct = 0
        if attack_breakdown:
            for attack_type, data in attack_breakdown.items():
                if data['count'] > dominant_attack_count:
                    dominant_attack = attack_type; dominant_attack_count = data['count']; dominant_attack_pct = data['percentage']
        recommendations = []
        if attack_pct > 0:
            recommendations.append("Lakukan investigasi lebih lanjut terhadap traffic yang terdeteksi sebagai serangan")
        if "DDoS" in attack_breakdown or "DoS" in attack_breakdown:
            recommendations.append("Pertimbangkan untuk mengaktifkan rate limiting dan DDoS protection")
        if "Port Scan" in attack_breakdown:
            recommendations.append("Review firewall rules dan tutup port yang tidak diperlukan")
        if "Brute Force" in attack_breakdown:
            recommendations.append("Terapkan account lockout policy dan gunakan strong authentication")
        if attack_pct == 0:
            recommendations.append("Traffic dalam kondisi normal, tetap monitor secara berkala")
        conclusion_text = f"""Dari total {total:,} flow jaringan yang dianalisis:\n- {summary['benign_count']:,} flow ({benign_pct:.2f}%) terdeteksi sebagai traffic NORMAL (BENIGN)\n- {summary['attack_count']:,} flow ({attack_pct:.2f}%) terdeteksi sebagai SERANGAN\n\nStatus Keamanan: {threat_level}\n{threat_description}""".strip()
        if dominant_attack:
            conclusion_text += f"\n\nJenis serangan dominan: {dominant_attack} ({dominant_attack_pct:.2f}% dari total traffic)"
        return {"threat_level": threat_level, "threat_color": threat_color, "threat_description": threat_description, "conclusion_text": conclusion_text, "dominant_attack": {"type": dominant_attack,"count": dominant_attack_count,"percentage": dominant_attack_pct} if dominant_attack else None, "recommendations": recommendations}

    def _generate_detailed_summary(self, all_results: List[Dict], processing_time: float) -> Dict:
        total_samples = len(all_results)
        prediction_counts = {}; confidence_sum = {}; confidence_values = {}
        for result in all_results:
            label = result['prediction']; conf = result['confidence']
            prediction_counts[label] = prediction_counts.get(label, 0) + 1
            confidence_sum[label] = confidence_sum.get(label, 0) + conf
            if label not in confidence_values: confidence_values[label] = []
            confidence_values[label].append(conf)
        benign_count = prediction_counts.get('BENIGN', 0)
        attack_count = total_samples - benign_count
        benign_pct = (benign_count / total_samples) * 100 if total_samples > 0 else 0
        attack_pct = (attack_count / total_samples) * 100 if total_samples > 0 else 0
        attack_breakdown = {}
        for label, count in prediction_counts.items():
            if label != 'BENIGN':
                pct = (count / total_samples) * 100
                avg_conf = confidence_sum[label] / count if count > 0 else 0
                min_conf = min(confidence_values[label]) if confidence_values[label] else 0
                max_conf = max(confidence_values[label]) if confidence_values[label] else 0
                attack_breakdown[label] = {"count": count, "percentage": round(pct, 2), "avg_confidence": round(avg_conf, 4), "min_confidence": round(min_conf, 4), "max_confidence": round(max_conf, 4)}
        attack_breakdown = dict(sorted(attack_breakdown.items(), key=lambda x: x[1]['count'], reverse=True))
        benign_stats = None
        if benign_count > 0:
            avg_conf = confidence_sum.get('BENIGN', 0) / benign_count
            benign_stats = {"count": benign_count, "percentage": round(benign_pct, 2), "avg_confidence": round(avg_conf, 4), "min_confidence": round(min(confidence_values.get('BENIGN', [0])), 4), "max_confidence": round(max(confidence_values.get('BENIGN', [0])), 4)}
        summary = {"total_samples": total_samples, "processing_time_seconds": round(processing_time, 3), "benign_count": benign_count, "benign_percentage": round(benign_pct, 2), "attack_count": attack_count, "attack_percentage": round(attack_pct, 2), "benign_stats": benign_stats, "attack_breakdown": attack_breakdown, "prediction_counts": prediction_counts}
        return summary

    def predict_dataset(self, df: pd.DataFrame, include_all_results: bool = False, batch_size: int = 1000) -> Dict:
        start_time = time.time()
        # Get required columns from preprocessor (45 features for transform)
        if not preprocessor.is_loaded:
            preprocessor.load()
        # Use feature_cols_fitted (45 features) for transformation, not final 10 features
        if hasattr(preprocessor, 'get_feature_names_fitted'):
            required_cols = preprocessor.get_feature_names_fitted()
        elif hasattr(preprocessor, 'feature_cols_fitted') and preprocessor.feature_cols_fitted:
            required_cols = preprocessor.feature_cols_fitted
        else:
            required_cols = preprocessor.get_feature_names()
        print("   🔄 Mapping columns...")
        df_mapped, mapping = SimpleColumnMapper.map_columns(df, required_cols)
        if mapping:
            print(f"   ✅ Mapped {len(mapping)} columns")
        # Check for missing columns (case-insensitive and canonicalized)
        missing = []
        for req_col in required_cols:
            req_canonical = SimpleColumnMapper._canonicalize_name(req_col)
            found = False
            for col in df_mapped.columns:
                col_canonical = SimpleColumnMapper._canonicalize_name(col)
                if col_canonical == req_canonical or col == req_col:
                    found = True
                    break
            if not found:
                missing.append(req_col)
        if missing:
            raise HTTPException(status_code=400, detail={"error": "Missing required columns", "missing_columns": missing, "your_columns": list(df.columns)[:20], "mapping_applied": mapping, "hint": "Check GET /api/v1/features for required column names"})
        df_selected = df_mapped[required_cols].copy()
        for c in df_selected.columns:
            if df_selected[c].dtype == object:
                df_selected[c] = df_selected[c].astype(str).str.replace(',', '', regex=False).str.strip().replace({'': None})
            df_selected[c] = pd.to_numeric(df_selected[c], errors='coerce')

        try:
            df_selected = preprocessor._recompute_rates(df_selected)
            df_selected = preprocessor._recompute_packet_length_stats(df_selected)
        except Exception as e:
            print("Warning: recompute helpers failed:", e)

        auto_fixes = []
        total_rows = len(df_selected)
        # Check if clip_map exists and is not empty
        has_clip_map = hasattr(preprocessor, 'clip_map') and preprocessor.clip_map
        if AUTO_FIX_EXTREME and has_clip_map:
            total_fixed_cells = 0
            per_feature_fixed = {}
            for c in required_cols:
                try:
                    if c not in preprocessor.clip_map: 
                        continue
                    # clip_map format: {feature: (q001, q999)} or {feature: [q001, q999]}
                    clip_data = preprocessor.clip_map[c]
                    if isinstance(clip_data, (list, tuple)) and len(clip_data) >= 2:
                        train_q999 = clip_data[1]
                    elif isinstance(clip_data, dict):
                        train_q999 = clip_data.get('q999', clip_data.get('q_999', None))
                    else:
                        train_q999 = clip_data
                    if train_q999 is None: continue
                    q999 = float(train_q999)
                    mask = df_selected[c] > (q999 * EXTREME_FACTOR)
                    if mask.any():
                        count = int(mask.sum())
                        total_fixed_cells += count
                        per_feature_fixed[c] = count
                        # record example max before fix using original values (if any)
                        example_max_before = float(df_mapped[c].dropna().max()) if c in df_mapped.columns else None
                        df_selected.loc[mask, c] = np.nan
                        auto_fixes.append({"feature": c, "fixed_count": count, "train_q999": q999, "example_max_before_fix": example_max_before})
                except Exception:
                    continue
            total_fixed_row_frac = total_fixed_cells / max(1, total_rows * len(required_cols))
            reject_reasons = []
            if total_fixed_row_frac > AUTO_FIX_TOTAL_ROW_FRAC_REJECT:
                reject_reasons.append({"reason": "too_many_fixed_cells_overall", "fraction": total_fixed_row_frac})
            for feat, cnt in per_feature_fixed.items():
                frac = (cnt / max(1, total_rows))
                if feat in ALLOW_AUTO_FIX:
                    # allowed feature: log but do not reject
                    print(f"Info: feature {feat} exceeded per-feature fix frac ({frac:.3f}) but is in ALLOW_AUTO_FIX; continuing.")
                    continue
                if frac > AUTO_FIX_PER_FEATURE_FRAC_REJECT:
                    reject_reasons.append({"reason": "too_many_fixed_in_feature", "feature": feat, "fixed_frac": frac})
            if reject_reasons:
                raise HTTPException(status_code=422, detail={"error": "Suspicious large values detected in uploaded file. Possible parsing/unit error. Auto-fix would be excessive.", "issues": auto_fixes, "reject_reasons": reject_reasons, "hint": "Check delimiter/thousands separators or ensure Flow Duration units match training (seconds).' "})

        if df_selected.isna().any().any():
            nan_cols = [c for c in required_cols if df_selected[c].isna().any()]
            has_medians = hasattr(preprocessor, 'medians') and preprocessor.medians
            if has_medians:
                print(f"Filling NaNs for columns: {nan_cols} using training medians")
                df_selected = preprocessor._fill_missing_with_medians(df_selected)
            else:
                raise HTTPException(status_code=400, detail={"error": "Input contains missing values and no training medians are available.", "na_columns": nan_cols, "hint": "Provide cleaned input or populate transform_meta.json with medians."})

        total_samples = len(df_selected)
        all_results = []
        aggregated_clip_reports: List[Dict] = []
        sample_fixes: List[Dict] = []

        print(f"   🔮 Predicting {total_samples} samples...")
        for i in range(0, total_samples, batch_size):
            batch_df = df_selected.iloc[i:i+batch_size]
            original_batch = batch_df.copy()
            try:
                recomputed_snapshot = preprocessor._recompute_rates(original_batch.copy())
                recomputed_snapshot = preprocessor._recompute_packet_length_stats(recomputed_snapshot)
            except Exception as e:
                recomputed_snapshot = original_batch.copy()
                print("Warning: recompute snapshot failed for batch:", e)
            for ridx, (orig_idx, orig_row) in enumerate(original_batch.iterrows()):
                if len(sample_fixes) >= 5:
                    break
                fixes = {}
                for col in required_cols:
                    before = orig_row.get(col, None)
                    after = recomputed_snapshot.at[orig_idx, col] if (col in recomputed_snapshot.columns and orig_idx in recomputed_snapshot.index) else before
                    try:
                        b = float(before) if not pd.isna(before) else None
                    except Exception:
                        b = None
                    try:
                        a = float(after) if (after is not None and not pd.isna(after)) else None
                    except Exception:
                        a = None
                    equal = False
                    if b is None and a is None:
                        equal = True
                    elif b is None and a is not None:
                        equal = False
                    elif a is None and b is not None:
                        equal = False
                    else:
                        if abs(b - a) <= max(1e-6, 1e-6 * abs(b)):
                            equal = True
                    if not equal:
                        fixes[col] = {"before": b, "after": a}
                if fixes:
                    sample_fixes.append({"index": int(orig_idx), "fixes": fixes})

            # ===== batch prediction and calibration =====
            # prepare transformed features for model
            X = preprocessor.transform(batch_df)

            # get raw results from model (should include 'probabilities' per sample)
            raw_batch_results = ids_model.predict_batch(X)

            # extract raw prob dicts in same order
            raw_prob_dicts = []
            for r in raw_batch_results:
                if isinstance(r, dict) and 'probabilities' in r and isinstance(r['probabilities'], dict):
                    raw_prob_dicts.append({k: float(v) for k, v in r['probabilities'].items()})
                else:
                    # fallback to prediction/confidence -> one-hot-like dict
                    lab = r.get('prediction') if isinstance(r, dict) else None
                    conf = float(r.get('confidence', 0.0)) if isinstance(r, dict) else 0.0
                    raw_prob_dicts.append({str(lab): conf} if lab is not None else {})

            # apply calibrators and renormalize (if any); returns list[dict]
            calibrated_prob_dicts = self._apply_calibration_and_renormalize(raw_prob_dicts)

            # compose final batch_results using calibrated probs and per-class thresholds
            batch_results = []
            for prob_dict in calibrated_prob_dicts:
                # determine predicted label and confidence
                if prob_dict:
                    pred_label, conf = max(prob_dict.items(), key=lambda kv: float(kv[1]))
                    conf = float(conf)
                else:
                    pred_label, conf = None, 0.0

                # determine is_attack using per-class threshold if available, else fallback ATTACK_CONF_THRESHOLD
                threshold_entry = self.per_class_thresholds.get(pred_label) if hasattr(self, 'per_class_thresholds') else None
                if threshold_entry and isinstance(threshold_entry, dict):
                    used_thresh = float(threshold_entry.get('threshold', ATTACK_CONF_THRESHOLD))
                else:
                    used_thresh = ATTACK_CONF_THRESHOLD

                is_attack = False
                if pred_label is not None:
                    if pred_label != 'BENIGN' and conf >= used_thresh:
                        is_attack = True
                    else:
                        is_attack = False

                batch_results.append({
                    "prediction": pred_label,
                    "confidence": conf,
                    "is_attack": is_attack,
                    "probabilities": prob_dict
                })
            # ===== end batch prediction and calibration =====

            clip_report = preprocessor.get_last_clip_report()
            aggregated_clip_reports.append({"batch_range": [int(i), int(min(i+batch_size, total_samples)-1)], "clip_report": clip_report})

            all_results.extend(batch_results)

        processing_time = time.time() - start_time
        summary = self._generate_detailed_summary(all_results, processing_time)
        conclusion = self._generate_conclusion(summary)
        self.last_dataset = df_mapped
        self.last_results = all_results
        self.last_mapping = mapping
        self.last_summary = summary

        # Check if label column exists for evaluation
        evaluation = None
        label_col = None
        label_candidates = ['Label', 'label', 'Label_collapsed', 'label_collapsed', 
                           'True Label', 'true_label', 'y_true', 'target', 'Label_encoded']
        
        for col in df_mapped.columns:
            if col in label_candidates:
                label_col = col
                break
        
        if label_col and label_col in df_mapped.columns:
            try:
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                y_true = df_mapped[label_col].astype(str).str.strip().values
                y_pred = [r['prediction'] for r in all_results]
                
                # Normalize labels (handle case sensitivity and whitespace)
                y_true_normalized = [str(label).strip() for label in y_true]
                y_pred_normalized = [str(label).strip() for label in y_pred]
                
                # Calculate accuracy
                accuracy = accuracy_score(y_true_normalized, y_pred_normalized) * 100
                
                # Per-class accuracy
                from collections import Counter
                per_class_accuracy = {}
                unique_labels = set(y_true_normalized + y_pred_normalized)
                
                for label in unique_labels:
                    mask = [y == label for y in y_true_normalized]
                    if sum(mask) > 0:
                        correct = sum([1 for i, pred in enumerate(y_pred_normalized) if mask[i] and pred == label])
                        per_class_accuracy[label] = (correct / sum(mask)) * 100
                
                # Classification report
                try:
                    class_report = classification_report(y_true_normalized, y_pred_normalized, 
                                                        output_dict=True, zero_division=0)
                except Exception:
                    class_report = None
                
                # Confusion matrix
                try:
                    cm = confusion_matrix(y_true_normalized, y_pred_normalized, 
                                        labels=sorted(unique_labels))
                    cm_list = cm.tolist()
                except Exception:
                    cm_list = None
                
                evaluation = {
                    "accuracy": round(accuracy, 2),
                    "per_class_accuracy": {k: round(v, 2) for k, v in per_class_accuracy.items()},
                    "classification_report": class_report,
                    "confusion_matrix": cm_list
                }
            except Exception as e:
                print(f"Warning: Could not calculate evaluation metrics: {e}")
                evaluation = None

        response = {'success': True, 'summary': summary, 'conclusion': conclusion, 'debug': {'clip_reports': aggregated_clip_reports, 'sample_fixes': sample_fixes, 'auto_fixes': auto_fixes}}
        if include_all_results:
            response['results'] = all_results
        else:
            response['sample_results'] = all_results[:10]
        
        if evaluation:
            response['evaluation'] = evaluation

        response = _sanitize_for_json(response)
        return response

    def get_results_as_dataframe(self) -> Optional[pd.DataFrame]:
        if self.last_results is None or self.last_dataset is None:
            return None
        results_df = pd.DataFrame(self.last_results)
        combined = self.last_dataset.copy()
        combined['predicted_label'] = results_df['prediction']
        combined['confidence'] = results_df['confidence']
        combined['is_attack'] = results_df['is_attack']
        return combined

    def export_results_csv(self) -> Optional[str]:
        """
        Export the last prediction results as CSV string.

        Returns:
            CSV string if results available, otherwise None
        """
        df = self.get_results_as_dataframe()
        if df is None:
            return None
        return df.to_csv(index=False)


dataset_handler = DatasetHandler()