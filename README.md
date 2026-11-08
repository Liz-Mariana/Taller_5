# Quinta lista de ejercicios
Introducción al Machine Learning

**Autora:** Mariana Lizarazo Buitrago y Karol Salcedo

## Descripción

Se predice el salario (`Salary`, en miles de dólares) de 263 jugadores de béisbol a partir de 19 estadísticas, usando regresión lineal con intercepto. Se comparan cuatro modelos según cómo seleccionan las variables:

1. Todas las variables.
2. Las 7 variables de mayor correlación de Pearson con `Salary`.
3. Las variables con valor p < 0,05 en la prueba t de la regresión completa.
4. Selección hacia adelante, eligiendo el número de variables con el menor riesgo de validación.

El riesgo se calcula con una función propia:

$$\hat{R}_S(h) = \frac{1}{|S|} \sum_{i \in S} \bigl(y_i - h(x_i)\bigr)^2$$

## Partición de los datos

Se usa `train_test_split` con `random_state=31`, en dos etapas:

1. 120 jugadores para desarrollo y 143 para prueba.
2. Los 120 de desarrollo se dividen en 50 para entrenamiento y 70 para validación.

La prueba se usa únicamente en el ejercicio 4.

## Estructura del repositorio

```
quinto_taller/
├── README.md
├── requirements.txt
├── data/
│   └── Datos_Hitters.csv
├── src/
│   ├── __init__.py
│   ├── data.py        # carga de datos y partición
│   ├── metrics.py     # riesgo cuadrático
│   └── modelos.py     # ajuste, evaluación y selección hacia adelante
└── notebooks/
    └── lista5.ipynb   # desarrollo de los ejercicios 1 a 4
```

## Cómo ejecutar

1. Instalar las dependencias:

```
pip install -r requirements.txt
```

2. Abrir `notebooks/lista5.ipynb`, reiniciar el kernel y ejecutar todas las celdas (**Run All**). Las rutas son relativas, por lo que el cuaderno debe ejecutarse desde la carpeta `notebooks/`.

## Resultados

| Modelo | Variables | Riesgo train | Riesgo validación | Riesgo prueba |
|---|---|---|---|---|
| Todas las variables | 19 | 68584.82 | 320423.19 | 394308.69 |
| Mayor correlación | CRBI, CRuns, CHmRun, CHits, CAtBat, CWalks, Years | 115487.25 | 133035.61 | 145923.29 |
| Pruebas t | AtBat, CWalks, Assists | 161221.36 | 140085.19 | 158772.47 |
| Selección hacia adelante | CRBI, Years, CRuns | 126982.32 | 130618.20 | 126939.60 |

- El modelo con todas las variables tiene el menor riesgo de entrenamiento, pero el mayor en validación y en prueba (sobreajuste con solo 50 jugadores de entrenamiento).
- El modelo elegido con validación fue **Selección hacia adelante** (3 variables: CRBI, Years y CRuns), y también resultó el de menor riesgo de prueba.