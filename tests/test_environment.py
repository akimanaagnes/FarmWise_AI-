
"""
FarmWise AI - Environment Tests

These tests verify that the essential Data Science
libraries are installed and working correctly.
"""

def test_numpy():
    import numpy as np
    assert np.array([1, 2, 3]).sum() == 6


def test_pandas():
    import pandas as pd
    data = pd.DataFrame({"crop": ["Maize", "Rice"]})
    assert len(data) == 2


def test_matplotlib():
    import matplotlib.pyplot as plt
    assert plt is not None


def test_seaborn():
    import seaborn as sns
    assert sns is not None


def test_scikit_learn():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    assert model is not None
