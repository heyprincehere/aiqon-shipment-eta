import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from features import FEATURES, validate_features
from generate_data import generate_shipments

def test_generator_is_reproducible():
    a = generate_shipments(12, seed=7)
    b = generate_shipments(12, seed=7)
    assert a.equals(b)

def test_generated_rows_have_expected_targets():
    df = generate_shipments(25, seed=3)
    assert len(df) == 25
    assert df["final_delay_hours"].notna().all()
    assert set(df["is_delayed_24h"].unique()).issubset({0, 1})

def test_feature_set_has_no_target_leakage():
    validate_features(FEATURES)
    with pytest.raises(ValueError):
        validate_features(FEATURES + ["final_delay_hours"])
