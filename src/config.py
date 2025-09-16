"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
