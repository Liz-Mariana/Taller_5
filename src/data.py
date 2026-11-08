import pandas as pd
from sklearn.model_selection import train_test_split

SEMILLA = 31

def cargar_datos(ruta):
    df = pd.read_csv(ruta, index_col="index")
    X = df.drop(columns="Salary")
    y = df["Salary"]
    return X, y

def particionar(X, y, semilla=SEMILLA):
    # Etapa 1: 120 desarrollo / 143 prueba
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, train_size=120, test_size=143, random_state=semilla
    )
    # Etapa 2: 50 entrenamiento / 70 validación
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, train_size=50, test_size=70, random_state=semilla
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test