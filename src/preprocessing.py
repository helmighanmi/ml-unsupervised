"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def get_scaler(name="robust"):
    """
    Return a scaler based on config name.
    Options: "standard", "minmax", "robust"
    """
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    elif name == "minmax":
        return MinMaxScaler()
    elif name == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {name}")
