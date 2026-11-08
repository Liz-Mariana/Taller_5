import numpy as np

def riesgo_cuadratico(y_real, y_pred):
    """Riesgo empírico cuadrático: (1/|S|) * sum (y_i - h(x_i))^2."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_real - y_pred) ** 2)