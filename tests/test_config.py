import yaml
from pathlib import Path

def test_config_has_robust_scaler():
    """Check that config.yaml defines RobustScaler as default scaler."""
    config_path = Path("config.yaml")
    assert config_path.exists(), "config.yaml is missing!"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    scaler = config.get("preprocessing", {}).get("scaler", None)
    assert scaler == "RobustScaler", f"Expected RobustScaler, but got {scaler}"
