"""Interactive Streamlit demo for shipment ETA delay risk."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parent
FEATURES = ["route", "route_distance_km", "planned_transit_hours", "planned_port_wait_hours", "planned_customs_hours", "departure_delay_hours", "port_arrival_delay_hours", "weather_severity", "port_congestion", "customs_complexity"]

st.set_page_config(page_title="Aiqon | Shipment ETA Intelligence", page_icon="🚢", layout="wide")
st.title("🚢 Shipment ETA Intelligence")
st.caption("Explainable delay forecasting · Synthetic research prototype · Prediction checkpoint: destination port arrival, before customs clearance")
model_path = ROOT / "models" / "eta_delay_regressor.joblib"
clf_path = ROOT / "models" / "delay_risk_classifier.joblib"
if not model_path.exists() or not clf_path.exists():
    st.warning("Models not found. Run `python src/generate_data.py` and `python src/train_model.py` first.")
    st.stop()
reg, clf = joblib.load(model_path), joblib.load(clf_path)
data = pd.read_csv(ROOT / "data" / "shipments.csv")
with st.sidebar:
    st.header("Shipment conditions")
    route = st.selectbox("Route", sorted(data.route.unique()))
    distance = st.number_input("Route distance (km)", 500.0, 10000.0, float(data.loc[data.route == route, "route_distance_km"].median()), 100.0)
    transit = st.number_input("Planned transit (hours)", 12.0, 1000.0, float(data.planned_transit_hours.median()))
    port_wait = st.number_input("Planned port wait (hours)", 0.0, 200.0, float(data.planned_port_wait_hours.median()))
    customs = st.number_input("Planned customs (hours)", 0.0, 100.0, float(data.planned_customs_hours.median()))
    dep_delay = st.number_input("Observed departure delay (hours)", -4.0, 100.0, 4.0)
    port_arrival_delay = st.number_input("Observed port-arrival delay (hours)", -4.0, 150.0, 8.0)
    weather = st.slider("Weather severity (0 = calm, 1 = severe)", 0.0, 1.0, 0.35)
    congestion = st.slider("Port congestion (0 = low, 1 = high)", 0.0, 1.0, 0.55)
    complexity = st.slider("Customs complexity (0 = low, 1 = high)", 0.0, 1.0, 0.4)
if st.button("Predict ETA and delay risk", type="primary"):
    X = pd.DataFrame([dict(route=route, route_distance_km=distance, planned_transit_hours=transit, planned_port_wait_hours=port_wait, planned_customs_hours=customs, departure_delay_hours=dep_delay, port_arrival_delay_hours=port_arrival_delay, weather_severity=weather, port_congestion=congestion, customs_complexity=complexity)])[FEATURES]
    delay = float(reg.predict(X)[0])
    prob = float(clf.predict_proba(X)[0][list(clf.named_steps["model"].classes_).index(1)]) if 1 in clf.named_steps["model"].classes_ else 0.0
    scheduled_hours = transit + port_wait + customs
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted final delay", f"{delay:.1f} hours")
    col2.metric("Risk of delay > 24 hours", f"{prob:.0%}")
    col3.metric("Estimated total journey", f"{scheduled_hours + delay:.1f} hours")
    st.subheader("What may drive the delay?")
    importance_path = ROOT / "models" / "permutation_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(5)
        st.bar_chart(imp.set_index("feature")["importance_mae_increase_hours"])
        st.caption("Global held-out permutation importance: indicates average model reliance, not proof of causation for this specific shipment.")
    st.info("This is a synthetic-data prototype. Outputs are not operational forecasts and should not be used for real shipping decisions.")

st.divider()
st.subheader("Dataset overview")
col1, col2, col3 = st.columns(3)
col1.metric("Synthetic journeys", f"{len(data):,}")
col2.metric("Routes simulated", f"{data.route.nunique()}")
col3.metric("Shipments delayed > 24h", f"{data.is_delayed_24h.mean():.0%}")
st.dataframe(data[["shipment_id", "route", "departure_delay_hours", "port_arrival_delay_hours", "port_processing_delay_hours", "customs_delay_hours", "final_delay_hours"]].head(20), use_container_width=True)
