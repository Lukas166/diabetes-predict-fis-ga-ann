from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
import streamlit as st

from .inference import risk_label


def render_prediction_card(title: str, score: float) -> None:
    clipped = float(np.clip(score, 0.0, 1.0))
    st.markdown(f"#### {title}")
    st.progress(clipped)
    st.metric("Probabilitas Risiko", f"{clipped * 100:.1f}%")
    st.markdown(f"<p class='risk-label'>Kategori: {risk_label(clipped)}</p>", unsafe_allow_html=True)


def plot_eda_charts(dataframe: pd.DataFrame) -> None:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### Distribusi Kelas Outcome")
        figure, axis = plt.subplots(figsize=(4.9, 3.2), facecolor="white")
        counts = dataframe["Outcome"].value_counts().sort_index()
        labels = ["Tidak Diabetes (0)", "Diabetes (1)"]
        axis.bar(labels, counts.values, color=["#9fb0df", "#2e499d"])
        axis.set_ylabel("Jumlah Sampel")
        axis.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(counts.values):
            axis.text(idx, value + 4, str(value), ha="center", fontsize=9)
        st.pyplot(figure, use_container_width=True)

    with col_right:
        st.markdown("##### Correlation Heatmap")
        figure, axis = plt.subplots(figsize=(5.1, 3.2), facecolor="white")
        corr = dataframe.corr(numeric_only=True)
        sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", linewidths=0.35, cbar=False, ax=axis)
        st.pyplot(figure, use_container_width=True)


def plot_performance_bars(mf_payload: dict) -> None:
    perf = (mf_payload or {}).get("performance", {})
    if not perf:
        st.info("Data performa model tidak tersedia pada mf_params.json.")
        return

    labels = ["FIS Manual", "FIS + GA", "ANFIS"]
    values = [
        float(perf.get("fis_manual_accuracy", 0.0)),
        float(perf.get("fis_ga_accuracy", 0.0)),
        float(perf.get("anfis_accuracy", 0.0)),
    ]

    figure, axis = plt.subplots(figsize=(5.0, 2.8), facecolor="white")
    bars = axis.bar(labels, values, color=["#647fbf", "#2e499d", "#0f766e"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Perbandingan Akurasi")
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            fontsize=8.5,
        )

    st.pyplot(figure, use_container_width=True)


def _mf_curve(universe: np.ndarray, points) -> np.ndarray:
    pts = np.array(points, dtype=float)
    if len(pts) == 3:
        return fuzz.trimf(universe, pts)
    return fuzz.trapmf(universe, pts)


def _gaussian_curve(x_norm: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma_safe = abs(float(sigma)) + 1e-6
    return np.exp(-0.5 * ((x_norm - float(mean)) / sigma_safe) ** 2)


def plot_shift_analysis(mf_payload: dict) -> None:
    manual_params = (mf_payload or {}).get("manual_mf_params", {})
    ga_params = (mf_payload or {}).get("ga_mf_params", {})
    anfis_params = (mf_payload or {}).get("anfis_mf_params", {})
    features = (mf_payload or {}).get("features", [])

    means = np.array(anfis_params.get("means", []), dtype=float)
    sigmas = np.array(anfis_params.get("sigmas", []), dtype=float)

    colors = {
        "low": "#2e499d",
        "medium": "#0f766e",
        "high": "#c2410c",
    }
    anfis_colors = ["#2e499d", "#0f766e", "#c2410c"]

    target_features: List[Tuple[str, str]] = [("Glucose", "Glucose"), ("BMI", "BMI"), ("Age", "Age")]

    figure, axes = plt.subplots(3, 3, figsize=(14.0, 8.8), facecolor="white")

    for row_idx, (feature_key, feature_name) in enumerate(target_features):
        for col_idx in range(3):
            axes[row_idx, col_idx].set_facecolor("white")

        if feature_key not in manual_params or feature_key not in ga_params:
            for col_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, "Data tidak tersedia", ha="center", va="center")
                axes[row_idx, col_idx].set_axis_off()
            continue

        manual_cfg = manual_params[feature_key]
        ga_cfg = ga_params[feature_key]

        f_min, f_max = manual_cfg.get("range", [0.0, 1.0])
        x = np.linspace(float(f_min), float(f_max), 500)
        span = max(float(f_max) - float(f_min), 1e-9)
        x_norm = (x - float(f_min)) / span

        ax_manual = axes[row_idx, 0]
        ax_ga = axes[row_idx, 1]
        ax_anfis = axes[row_idx, 2]

        for set_name in ["low", "medium", "high"]:
            color = colors[set_name]
            ax_manual.plot(x, _mf_curve(x, manual_cfg[set_name]), color=color, linewidth=2, label=set_name.capitalize())
            ax_ga.plot(x, _mf_curve(x, ga_cfg[set_name]), color=color, linewidth=2, linestyle="--", label=set_name.capitalize())

        ax_manual.set_title(f"FIS Manual - {feature_name}", fontsize=10.5, fontweight="bold")
        ax_ga.set_title(f"FIS + GA - {feature_name}", fontsize=10.5, fontweight="bold")

        if feature_key in features and means.size > 0 and sigmas.size > 0:
            feat_idx = features.index(feature_key)
            if feat_idx < means.shape[0] and feat_idx < sigmas.shape[0]:
                n_mf = min(3, means.shape[1], sigmas.shape[1])
                for i in range(n_mf):
                    curve = _gaussian_curve(x_norm, means[feat_idx, i], sigmas[feat_idx, i])
                    ax_anfis.plot(x, curve, color=anfis_colors[i], linewidth=2, label=f"MF {i + 1}")
                ax_anfis.set_title(f"ANFIS Gaussian - {feature_name}", fontsize=10.5, fontweight="bold")
            else:
                ax_anfis.text(0.5, 0.5, "Parameter ANFIS tidak valid", ha="center", va="center")
                ax_anfis.set_axis_off()
        else:
            ax_anfis.text(0.5, 0.5, "Parameter ANFIS tidak tersedia", ha="center", va="center")
            ax_anfis.set_axis_off()

        for ax in [ax_manual, ax_ga, ax_anfis]:
            if ax.axison:
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel(feature_name)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)

        ax_manual.set_ylabel("Membership")

    plt.tight_layout()
    st.pyplot(figure, use_container_width=True)
