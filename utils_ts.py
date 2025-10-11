import numpy as np
import pandas as pd
import hashlib

def deterministic_random(seed_str: str):
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    return rng

def synthesize_history(current_value, highest_value, length=12, trend=0.02, noise_scale=0.05, seed_str=''):
    """
    Create a synthetic historical series for a player given current and highest values.
    This is a deterministic generator (seeded by player) to allow consistent plots.
    """
    rng = deterministic_random(seed_str)
    # start around current_value but simulate past ups/downs toward highest_value
    base = float(current_value) if current_value and current_value>0 else float(highest_value or 1000000)
    highest = float(highest_value) if highest_value and highest_value>0 else base
    # a simple upward/downward trend anchored at base and highest
    steps = np.arange(-length+1, 1)
    trend_component = (highest - base) * (np.tanh(0.2*steps) + 1)/2  # smooth shape
    noise = rng.normal(scale=noise_scale*base, size=length)
    series = base + trend_component + noise
    series = np.maximum(series, 0)
    return series.tolist()

def synthesize_sentiment(name, length=12):
    """
    Deterministic synthetic sentiment time series in range [-1,1] seeded by name.
    """
    rng = deterministic_random(name)
    # base sentiment depends on hash parity, with slow variations
    base = (rng.rand() - 0.5) * 0.4
    noise = rng.normal(scale=0.2, size=length)
    trend = np.linspace(-0.1, 0.1, length) * (rng.rand()-0.5)
    series = base + noise + trend
    series = np.clip(series, -1, 1)
    return series.tolist()
