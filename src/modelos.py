from sklearn.linear_model import LinearRegression
from src.metrics import riesgo_cuadratico


def ajustar_y_evaluar(variables, X_tr, y_tr, X_val, y_val):
    """Ajusta regresión lineal con intercepto en entrenamiento usando
    solo `variables` y devuelve (modelo, riesgo_train, riesgo_val)."""
    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(X_tr[variables], y_tr)
    r_tr = riesgo_cuadratico(y_tr, modelo.predict(X_tr[variables]))
    r_val = riesgo_cuadratico(y_val, modelo.predict(X_val[variables]))
    return modelo, r_tr, r_val

def seleccion_adelante(X_tr, y_tr, X_val, y_val):
    """Selección hacia adelante: en cada paso agrega la variable que más
    reduce el riesgo de entrenamiento. Devuelve lista de pasos."""
    elegidas = []
    restantes = list(X_tr.columns)
    pasos = []
    while restantes:
        mejor = min(
            restantes,
            key=lambda c: ajustar_y_evaluar(elegidas + [c], X_tr, y_tr, X_val, y_val)[1],
        )
        elegidas.append(mejor)
        restantes.remove(mejor)
        _, r_tr, r_val = ajustar_y_evaluar(elegidas, X_tr, y_tr, X_val, y_val)
        pasos.append({
            "N° variables": len(elegidas),
            "Variable agregada": mejor,
            "Riesgo train": r_tr,
            "Riesgo validación": r_val,
            "variables": list(elegidas),
        })
    return pasos