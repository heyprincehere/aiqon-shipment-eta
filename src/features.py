"""Feature definitions. Keep post-checkpoint outcomes out of model inputs."""
FEATURES = [
    "route", "route_distance_km", "planned_transit_hours", "planned_port_wait_hours",
    "planned_customs_hours", "departure_delay_hours", "port_arrival_delay_hours",
    "weather_severity", "port_congestion", "customs_complexity",
]
REGRESSION_TARGET = "final_delay_hours"
CLASSIFICATION_TARGET = "is_delayed_24h"
FORBIDDEN_FEATURES = {
    "actual_customs_clearance_utc", "actual_final_arrival_utc", "customs_delay_hours",
    "port_processing_delay_hours", "final_delay_hours", "is_delayed_24h",
}

def validate_features(feature_names):
    leaked = set(feature_names) & FORBIDDEN_FEATURES
    if leaked:
        raise ValueError(f"Target leakage: forbidden features included: {sorted(leaked)}")
