"""Generate local and global explanations; SHAP is optional and gracefully falls back."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from features import FEATURES
ROOT = Path(__file__).resolve().parents[1]

def make_plots():
    data = pd.read_csv(ROOT / "data" / "shipments.csv")
    importance = pd.read_csv(ROOT / "models" / "permutation_importance.csv")
    pred = pd.read_csv(ROOT / "models" / "test_predictions.csv")
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    top = importance.head(10).sort_values("importance_mae_increase_hours")
    plt.barh(top["feature"], top["importance_mae_increase_hours"])
    plt.xlabel("Increase in test MAE when shuffled (hours)")
    plt.title("Held-out permutation importance")
    plt.tight_layout(); plt.savefig(out / "feature_importance.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(pred["final_delay_hours"], pred["predicted_delay_hours"], alpha=.7)
    lo = min(pred["final_delay_hours"].min(), pred["predicted_delay_hours"].min())
    hi = max(pred["final_delay_hours"].max(), pred["predicted_delay_hours"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual final delay (hours)"); plt.ylabel("Predicted delay (hours)")
    plt.title("Held-out ETA delay predictions")
    plt.tight_layout(); plt.savefig(out / "actual_vs_predicted.png", dpi=160); plt.close()
    # SHAP TreeExplainer on the underlying forest after one-hot encoding can be added when available.
    try:
        import shap
        model = joblib.load(ROOT / "models" / "eta_delay_regressor.joblib")
        X = data[FEATURES].sample(min(80, len(data)), random_state=42)
        transformed = model.named_steps["preprocess"].transform(X)
        names = model.named_steps["preprocess"].get_feature_names_out()
        explainer = shap.TreeExplainer(model.named_steps["model"])
        values = explainer.shap_values(transformed)
        shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()
        (out / "shap_status.txt").write_text("SHAP summary generated. Encoded feature names are used.")
    except Exception as exc:
        (out / "shap_status.txt").write_text(f"SHAP plot not generated: {type(exc).__name__}: {exc}. Use held-out permutation importance as the validated explanation.")
    print(f"Plots written to {out}")

if __name__ == "__main__":
    make_plots()
