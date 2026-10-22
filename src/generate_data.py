"""Generate a reproducible synthetic shipment dataset with causal delay propagation."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

def generate_shipments(n: int = 600, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    routes = ["Shanghai-Mumbai", "Singapore-Chennai", "Rotterdam-Mundra", "Dubai-Nhava Sheva", "Busan-Visakhapatnam"]
    route_distance = {"Shanghai-Mumbai": 5200, "Singapore-Chennai": 2900, "Rotterdam-Mundra": 6800, "Dubai-Nhava Sheva": 1000, "Busan-Visakhapatnam": 4300}
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n):
        route = routes[int(rng.integers(0, len(routes)))]
        distance = route_distance[route] * rng.normal(1, .035)
        planned_departure = base + pd.Timedelta(days=int(i * 1.7))
        planned_transit = distance / rng.uniform(330, 470) + rng.normal(12, 5)
        planned_port_wait = max(3, rng.normal(18, 7))
        planned_customs = max(2, rng.normal(12, 4))
        scheduled_arrival = planned_departure + pd.Timedelta(hours=planned_transit + planned_port_wait + planned_customs)
        congestion = float(np.clip(rng.beta(2.1, 2.0), 0, 1))
        weather = float(np.clip(rng.beta(1.5, 3.0), 0, 1))
        customs_complexity = float(np.clip(rng.beta(2, 2), 0, 1))
        departure_delay = max(-4, rng.normal(2 + 5*congestion, 7))
        sea_weather_delay = max(-2, rng.normal(2 + 11*weather, 4))
        port_arrival_delay = departure_delay * .58 + sea_weather_delay + max(0, rng.normal(3*congestion, 2))
        # Queueing interaction: congestion makes upstream delays more likely to carry forward.
        port_delay = max(0, rng.normal(3 + 12*congestion + .14*max(0, port_arrival_delay), 4))
        customs_delay = max(0, rng.normal(1 + 8*customs_complexity + 4*congestion + .12*port_delay, 3))
        final_delay = departure_delay*.25 + sea_weather_delay + port_delay + customs_delay + rng.normal(0, 3)
        actual_departure = planned_departure + pd.Timedelta(hours=departure_delay)
        actual_port_arrival = planned_departure + pd.Timedelta(hours=planned_transit + port_arrival_delay)
        actual_customs_clearance = actual_port_arrival + pd.Timedelta(hours=planned_port_wait + port_delay + planned_customs + customs_delay)
        actual_final_arrival = scheduled_arrival + pd.Timedelta(hours=final_delay)
        rows.append({
            "shipment_id": f"SHP-{i+1:04d}", "route": route, "route_distance_km": round(distance, 1),
            "planned_departure_utc": planned_departure.isoformat(), "scheduled_arrival_utc": scheduled_arrival.isoformat(),
            "actual_departure_utc": actual_departure.isoformat(), "actual_port_arrival_utc": actual_port_arrival.isoformat(),
            "actual_customs_clearance_utc": actual_customs_clearance.isoformat(), "actual_final_arrival_utc": actual_final_arrival.isoformat(),
            "planned_transit_hours": round(planned_transit, 2), "planned_port_wait_hours": round(planned_port_wait, 2),
            "planned_customs_hours": round(planned_customs, 2), "departure_delay_hours": round(departure_delay, 2),
            "port_arrival_delay_hours": round(port_arrival_delay, 2), "port_processing_delay_hours": round(port_delay, 2),
            "customs_delay_hours": round(customs_delay, 2), "weather_severity": round(weather, 3),
            "port_congestion": round(congestion, 3), "customs_complexity": round(customs_complexity, 3),
            "final_delay_hours": round(final_delay, 2), "is_delayed_24h": int(final_delay > 24),
            "prediction_checkpoint": "after_destination_port_arrival_before_customs_clearance",
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "shipments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_shipments()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} shipments to {out}")
