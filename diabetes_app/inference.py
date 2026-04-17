from typing import Dict, Optional, Tuple

import numpy as np
import skfuzzy as fuzz

from .config import CORE_FIS_FEATURES, DEFAULT_FEATURE_ORDER, RISK_RANGE


_ANFIS_RUNTIME_CACHE = {}


def _mf_curve(universe: np.ndarray, points) -> np.ndarray:
    pts = np.array(points, dtype=float)
    if len(pts) == 3:
        return fuzz.trimf(universe, pts)
    if len(pts) == 4:
        return fuzz.trapmf(universe, pts)
    # Fallback for malformed points: interpolate to triangular shape.
    pts = np.sort(pts)
    if len(pts) < 3:
        pts = np.array([pts[0], np.mean(pts), pts[-1]], dtype=float)
    return fuzz.trimf(universe, [pts[0], pts[len(pts) // 2], pts[-1]])


def _feature_degrees(feature_cfg: dict, value: float) -> Dict[str, float]:
    f_min, f_max = feature_cfg.get("range", [0.0, 1.0])
    universe = np.linspace(float(f_min), float(f_max), 500)

    degrees = {}
    for set_name in ["low", "medium", "high"]:
        curve = _mf_curve(universe, feature_cfg[set_name])
        degrees[set_name] = float(fuzz.interp_membership(universe, curve, value))

    return degrees


def _fuzzy_core_score(params: dict, inputs: Dict[str, float]) -> float:
    glucose = _feature_degrees(params["Glucose"], float(inputs["Glucose"]))
    bmi = _feature_degrees(params["BMI"], float(inputs["BMI"]))
    age = _feature_degrees(params["Age"], float(inputs["Age"]))

    risk_low = fuzz.trapmf(RISK_RANGE, [0.0, 0.0, 0.25, 0.45])
    risk_medium = fuzz.trapmf(RISK_RANGE, [0.35, 0.5, 0.5, 0.65])
    risk_high = fuzz.trapmf(RISK_RANGE, [0.55, 0.75, 1.0, 1.0])

    rules = [
        ("high", min(glucose["high"], 1.0)),
        ("high", min(glucose["high"], bmi["high"])),
        ("high", min(glucose["high"], age["high"])),
        ("high", min(glucose["medium"], bmi["high"], age["high"])),
        ("medium", min(glucose["medium"], bmi["medium"])),
        ("medium", min(glucose["medium"], age["medium"])),
        ("medium", min(glucose["low"], bmi["high"])),
        ("medium", min(glucose["medium"], bmi["low"])),
        ("low", min(glucose["low"], bmi["low"])),
        ("low", min(glucose["low"], age["low"], bmi["low"])),
    ]

    agg_high = np.zeros_like(RISK_RANGE)
    agg_medium = np.zeros_like(RISK_RANGE)
    agg_low = np.zeros_like(RISK_RANGE)

    for outcome, strength in rules:
        if outcome == "high":
            agg_high = np.fmax(agg_high, np.fmin(strength, risk_high))
        elif outcome == "medium":
            agg_medium = np.fmax(agg_medium, np.fmin(strength, risk_medium))
        else:
            agg_low = np.fmax(agg_low, np.fmin(strength, risk_low))

    aggregated = np.fmax(agg_high, np.fmax(agg_medium, agg_low))

    try:
        score = float(fuzz.defuzz(RISK_RANGE, aggregated, "centroid"))
    except Exception:
        score = 0.5

    return float(np.clip(score, 0.0, 1.0))


def fuzzy_predict_risk(inputs: Dict[str, float], mf_payload: dict, mode: str) -> Tuple[Optional[float], Optional[str]]:
    if not mf_payload:
        return None, "Parameter MF belum tersedia."

    if mode == "manual":
        params = mf_payload.get("manual_mf_params")
    elif mode == "ga":
        params = mf_payload.get("ga_mf_params")
    else:
        return None, f"Mode fuzzy tidak dikenali: {mode}"

    if not params:
        return None, f"Parameter MF untuk mode {mode} tidak ditemukan."

    missing = [name for name in CORE_FIS_FEATURES if name not in params]
    if missing:
        return None, f"Parameter MF fitur inti hilang: {', '.join(missing)}"

    try:
        score = _fuzzy_core_score(params, inputs)
        return score, None
    except Exception as exc:
        return None, f"Gagal melakukan inferensi FIS {mode}: {exc}"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _extract_probability(raw_output) -> float:
    arr = np.asarray(raw_output)

    if arr.ndim == 0:
        prob = float(arr)
    elif arr.ndim == 1:
        prob = float(arr[0])
    else:
        if arr.shape[1] >= 2:
            prob = float(arr[0, 1])
        else:
            prob = float(arr[0, 0])

    if prob < 0.0 or prob > 1.0:
        prob = float(_sigmoid(np.array([prob]))[0])

    return float(np.clip(prob, 0.0, 1.0))


def _build_runtime_anfis(n_features: int, n_mf: int):
    import torch
    import torch.nn as nn

    class RuntimeANFISLayer(nn.Module):
        def __init__(self, n_features_: int, n_mf_: int = 3):
            super().__init__()
            self.n_features = n_features_
            self.n_mf = n_mf_

            means_init = torch.linspace(0.1, 0.9, n_mf_).unsqueeze(0).repeat(n_features_, 1)
            sigmas_init = torch.full((n_features_, n_mf_), 0.3)

            self.means = nn.Parameter(means_init)
            self.sigmas = nn.Parameter(sigmas_init)
            self.output_layer = nn.Linear(n_features_ * n_mf_, 1)

        def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
            x_exp = x.unsqueeze(-1)
            mu = torch.exp(-0.5 * ((x_exp - self.means) / (self.sigmas.abs() + 1e-6)) ** 2)
            return mu

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mu = self.fuzzify(x)
            flat = mu.reshape(mu.size(0), -1)
            out = torch.sigmoid(self.output_layer(flat))
            return out

    return RuntimeANFISLayer(n_features, n_mf)


def _predict_anfis_from_bundle(inputs: Dict[str, float], model_bundle: dict, feature_order) -> Tuple[Optional[float], Optional[str]]:
    try:
        import torch
    except Exception as exc:
        return None, f"PyTorch tidak tersedia untuk inferensi ANFIS: {exc}"

    state_dict = model_bundle.get("model_state_dict")
    if state_dict is None:
        return None, "ANFIS bundle tidak memiliki model_state_dict."

    n_features = int(model_bundle.get("n_features", len(feature_order)))
    n_mf = int(model_bundle.get("n_mf", 3))
    scaler = model_bundle.get("scaler")

    try:
        raw_x = np.array([[float(inputs.get(name, 0.0)) for name in feature_order]], dtype=np.float32)
        if scaler is not None and hasattr(scaler, "transform"):
            x = scaler.transform(raw_x).astype(np.float32)
        else:
            x = raw_x

        cache_key = (id(state_dict), n_features, n_mf)
        model = _ANFIS_RUNTIME_CACHE.get(cache_key)
        if model is None:
            model = _build_runtime_anfis(n_features=n_features, n_mf=n_mf)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            _ANFIS_RUNTIME_CACHE[cache_key] = model

        with torch.no_grad():
            tensor_x = torch.tensor(x, dtype=torch.float32)
            raw = model(tensor_x)

        return _extract_probability(raw), None
    except Exception as exc:
        return None, f"Inferensi ANFIS dari model_state_dict gagal: {exc}"


def _normalize_inputs(inputs: Dict[str, float], reference_params: dict, feature_order) -> np.ndarray:
    values = []

    for feature in feature_order:
        feature_cfg = reference_params.get(feature)
        raw_value = float(inputs.get(feature, 0.0))

        if feature_cfg and "range" in feature_cfg:
            lo, hi = feature_cfg["range"]
            span = max(float(hi) - float(lo), 1e-9)
            norm_value = (raw_value - float(lo)) / span
        else:
            norm_value = raw_value

        values.append(norm_value)

    return np.array(values, dtype=np.float32).reshape(1, -1)


def _anfis_surrogate_score(inputs: Dict[str, float], mf_payload: dict) -> Tuple[Optional[float], Optional[str]]:
    anfis_params = mf_payload.get("anfis_mf_params")
    manual_params = mf_payload.get("manual_mf_params")
    feature_order = mf_payload.get("features", DEFAULT_FEATURE_ORDER)

    if not anfis_params or not manual_params:
        return None, "Parameter ANFIS pada mf_params.json tidak lengkap."

    means = np.array(anfis_params.get("means", []), dtype=float)
    sigmas = np.abs(np.array(anfis_params.get("sigmas", []), dtype=float))

    if means.size == 0 or sigmas.size == 0:
        return None, "Means/sigmas ANFIS tidak ditemukan."

    if means.shape != sigmas.shape:
        return None, "Dimensi means dan sigmas ANFIS tidak cocok."

    if means.shape[0] != len(feature_order):
        return None, "Dimensi parameter ANFIS tidak sesuai jumlah fitur."

    x = _normalize_inputs(inputs, manual_params, feature_order).reshape(-1)

    eps = 1e-6
    sigmas = np.maximum(sigmas, eps)
    gauss = np.exp(-0.5 * ((x[:, None] - means) / sigmas) ** 2)

    term_scores = np.array([0.1, 0.5, 0.9], dtype=float)
    weighted = gauss @ term_scores
    normalizer = np.maximum(np.sum(gauss, axis=1), eps)
    feature_scores = weighted / normalizer

    score = float(np.mean(feature_scores))
    return float(np.clip(score, 0.0, 1.0)), None


def anfis_predict_risk(
    inputs: Dict[str, float], mf_payload: dict, anfis_model=None
) -> Tuple[Optional[float], Optional[str]]:
    if not mf_payload:
        return None, "Parameter MF belum tersedia."

    manual_params = mf_payload.get("manual_mf_params", {})
    feature_order = mf_payload.get("features", DEFAULT_FEATURE_ORDER)

    x = _normalize_inputs(inputs, manual_params, feature_order)

    if anfis_model is not None:
        try:
            # Handle bundle format from saved PyTorch checkpoint.
            if isinstance(anfis_model, dict) and "model_state_dict" in anfis_model:
                bundle_score, bundle_err = _predict_anfis_from_bundle(inputs, anfis_model, feature_order)
                if bundle_score is not None:
                    return bundle_score, None

                surrogate_score, surrogate_err = _anfis_surrogate_score(inputs, mf_payload)
                if surrogate_score is not None:
                    return (
                        surrogate_score,
                        f"Prediksi ANFIS bundle gagal ({bundle_err}); skor memakai surrogate ANFIS dari parameter means/sigmas.",
                    )
                return None, f"Prediksi ANFIS bundle gagal: {bundle_err}; fallback juga gagal: {surrogate_err}"

            if hasattr(anfis_model, "predict_proba"):
                raw = anfis_model.predict_proba(x)
            elif hasattr(anfis_model, "predict"):
                raw = anfis_model.predict(x)
            else:
                try:
                    raw = anfis_model(x)
                except Exception:
                    import torch

                    tensor_x = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        raw = anfis_model(tensor_x)

            return _extract_probability(raw), None
        except Exception as model_exc:
            surrogate_score, surrogate_err = _anfis_surrogate_score(inputs, mf_payload)
            if surrogate_score is not None:
                return (
                    surrogate_score,
                    f"Model ANFIS PKL gagal dipakai ({model_exc}); skor memakai surrogate ANFIS dari parameter means/sigmas.",
                )
            return None, f"Prediksi ANFIS gagal: {model_exc}; fallback juga gagal: {surrogate_err}"

    surrogate_score, surrogate_err = _anfis_surrogate_score(inputs, mf_payload)
    if surrogate_score is not None:
        return surrogate_score, "Model ANFIS PKL tidak tersedia; skor memakai surrogate ANFIS dari parameter means/sigmas."

    return None, surrogate_err


def risk_label(score: float) -> str:
    if score < 0.35:
        return "Rendah"
    if score < 0.65:
        return "Sedang"
    return "Tinggi"
