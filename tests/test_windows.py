import numpy as np
from src.train import make_windows

def test_window_shapes_and_order():
    values = np.arange(8, dtype=float).reshape(-1, 1)
    X, y = make_windows(values, 3)
    assert X.shape == (5, 3, 1)
    assert y.tolist() == [3, 4, 5, 6, 7]
    assert X[0, :, 0].tolist() == [0, 1, 2]
