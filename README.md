# Curve Extrapolation for Early Stopping in Bayesian Optimization

## Descripción

Este proyecto desarrolla y evalúa un criterio de parada anticipada basado en la extrapolación de curvas de aprendizaje, integrado en un proceso de optimización bayesiana de hiperparámetros.

El objetivo principal es reducir el coste computacional asociado al entrenamiento de modelos de aprendizaje automático durante la búsqueda de hiperparámetros, evitando continuar aquellos entrenamientos cuyo rendimiento futuro estimado no parece suficientemente prometedor.

Para ello, se utiliza **XGBoost** como algoritmo de aprendizaje automático y **Optuna** como framework de optimización de hiperparámetros. Durante cada ensayo de optimización se registra la evolución de la métrica de validación y se ajusta una función matemática a una parte inicial de la curva. A partir de este ajuste se estima su comportamiento futuro y se decide si resulta conveniente continuar el entrenamiento.

El proyecto compara el método propuesto con dos estrategias de referencia:

- Entrenamiento sin parada anticipada.
- Entrenamiento con parada anticipada convencional.
- Entrenamiento con parada anticipada basada en extrapolación de curvas.

---

## Objetivos

Los principales objetivos del proyecto son:

- Estudiar la aplicación de la extrapolación de curvas de aprendizaje a la optimización de hiperparámetros.
- Comparar distintas familias de funciones matemáticas para representar la evolución de las métricas de validación.
- Desarrollar un criterio de parada anticipada basado en el rendimiento futuro estimado.
- Evaluar la capacidad del método para identificar entrenamientos que no resulta conveniente continuar.
- Analizar la reducción del tiempo de ejecución y del número de rondas de entrenamiento.
- Comparar el método propuesto con estrategias convencionales de parada anticipada y con el entrenamiento completo.

## Funcionamiento del método

El método propuesto utiliza la evolución de la métrica de validación durante las rondas de entrenamiento de XGBoost.

El procedimiento general es el siguiente:

1. Optuna genera una configuración de hiperparámetros para un nuevo ensayo.
2. Se inicia el entrenamiento de XGBoost y se registra la métrica de validación en cada ronda.
3. Una vez observada una parte inicial de la curva, aproximadamente el 20 % de las rondas de entrenamiento, se ajusta una función matemática a los valores registrados.
4. La función ajustada se utiliza para extrapolar la evolución esperada de la métrica durante las rondas restantes.
5. Se estima el mejor rendimiento que podría alcanzar el ensayo, teniendo en cuenta tanto los valores observados como los valores predichos.
6. La estimación se compara con el mejor resultado obtenido hasta ese momento en el proceso de optimización.
7. Si el rendimiento futuro estimado no permite superar el mejor resultado conocido, se interrumpe el entrenamiento del ensayo.


Para un problema de minimización, el criterio de parada se basa en comparar el mejor valor futuro estimado con el mejor valor obtenido anteriormente.
Si se cumple esta condición, el entrenamiento se interrumpe.

---

## Comparación de modelos de extrapolación

Además de evaluar el criterio de parada, se realiza un experimento para comparar distintas familias de funciones matemáticas aplicadas a la extrapolación de curvas de aprendizaje.

Se estudian las siguientes funciones:

- Exponencial doble (*Double Exponential*).
- Exponencial (*Exponential*).
- Ley de potencia (*Power Law*).
- Logarítmica (*Logarithmic*).
- Racional (*Rational*).
- Racional general (*General Rational*).
- Weibull.
- Exponencial estirada (*Stretched Exponential*).
- Gompertz.
- Logística (*Logistic*).

En este experimento se utiliza el conjunto de datos Wine Quality Red y se realizan 50 ensayos de optimización. Para cada ensayo se utiliza el 20 % inicial de la curva para ajustar las funciones y se compara la predicción del valor de la última ronda con el valor real observado.

La comparación considera tres indicadores:

- Error absoluto medio de las predicciones finales.
- Desviación estándar del error.
- Proporción de ajustes realizados correctamente.

---

## Configuración experimental

Los experimentos principales comparan las tres estrategias de entrenamiento descritas anteriormente.

### Configuración general

| Parámetro | Configuración |
|---|---|
| Algoritmo de aprendizaje | XGBoost |
| Framework de optimización | Optuna |
| Muestreador | GPSampler |
| Número de ensayos por semilla | 35 |
| Número de semillas | 15 |
| Total de ensayos por estrategia y conjunto de datos | 525 |
| División de los datos | 80 % entrenamiento / 20 % validación |
| Porcentaje inicial observado para extrapolación | 20 % |

Las semillas utilizadas en los experimentos principales son:

```python
[1, 2, 3, 4, 5, 10, 20, 40, 60, 100, 200, 400, 600, 800, 1000]
```

### Espacio de búsqueda de hiperparámetros

| Hiperparámetro | Rango |
|---|---|
| `max_depth` | 3–10 |
| `learning_rate` | 0.001–0.3 |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `min_child_weight` | 1–10 |
| `gamma` | 0–5 |
| `num_boost_round` | 100–1000 |

El espacio de búsqueda se utiliza para explorar distintas configuraciones del modelo y evaluar su rendimiento sobre el conjunto de validación.

### Conjuntos de datos

Los experimentos principales incluyen los siguientes conjuntos de datos:

- `loan_data`
- `stroke_dataset`
- `wine_quality_red`

El experimento específico de comparación de funciones de extrapolación utiliza Wine Quality Red.

---

## Resultados principales

Los resultados de los experimentos principales muestran que el método basado en extrapolación puede reducir el tiempo de entrenamiento, aunque existe un compromiso entre el ahorro computacional y el rendimiento predictivo.

### Comparación del rendimiento y el tiempo

Los siguientes valores representan las medias obtenidas en los experimentos:

| Conjunto de datos | Estrategia | MAE medio | Tiempo medio |
|---|---|---:|---:|
| loan_data | Sin parada anticipada | 0.144736 | 60.06 s |
| loan_data | Parada anticipada convencional | 0.144565 | 36.16 s |
| loan_data | Extrapolación | 0.145698 | 22.23 s |
| stroke_dataset | Sin parada anticipada | 0.154084 | 8.94 s |
| stroke_dataset | Parada anticipada convencional | 0.154342 | 4.94 s |
| stroke_dataset | Extrapolación | 0.153983 | 5.52 s |
| wine_quality_red | Sin parada anticipada | 0.363403 | 14.40 s |
| wine_quality_red | Parada anticipada convencional | 0.362962 | 13.48 s |
| wine_quality_red | Extrapolación | 0.365739 | 9.14 s |

### Ahorro de tiempo

En comparación con el entrenamiento sin parada anticipada, el método de extrapolación obtiene los siguientes ahorros de tiempo:

| Conjunto de datos | Ahorro de tiempo |
|---|---:|
| loan_data | 62.64 % |
| stroke_dataset | 36.76 % |
| wine_quality_red | 34.17 % |

Los resultados muestran que el ahorro computacional depende del conjunto de datos y de la evolución de las curvas de aprendizaje. En particular, el método obtiene el mayor ahorro en `loan_data`.

No obstante, la extrapolación no presenta el mejor MAE en todos los conjuntos de datos. Por tanto, los resultados deben interpretarse como un compromiso entre eficiencia computacional y calidad predictiva, y no como una mejora universal frente a las estrategias de referencia.

---

## Tecnologías utilizadas

El proyecto utiliza principalmente las siguientes herramientas y bibliotecas:

- **Python:** lenguaje de programación utilizado para implementar los experimentos.
- **XGBoost:** biblioteca de aprendizaje automático utilizada para entrenar los modelos.
- **Optuna:** framework empleado para la optimización bayesiana de hiperparámetros.
- **SciPy:** utilizado para el ajuste no lineal de las funciones matemáticas mediante `curve_fit`.
- **NumPy:** operaciones numéricas y manipulación de arrays.
- **Pandas:** procesamiento y análisis de datos.
- **Scikit-learn:** preparación de datos y herramientas de evaluación.
- **Matplotlib:** representación gráfica de las curvas de aprendizaje y los resultados.

---

Descripción de los directorios:

- `data/`: conjuntos de datos utilizados en los experimentos.
- `src/`: implementación del método de extrapolación y los experimentos.
- `results/`: resultados numéricos generados durante las ejecuciones.
- `README.md`: documentación general del repositorio.

---

## Limitaciones

El método propuesto presenta algunas limitaciones que deben tenerse en cuenta:

- La precisión de la extrapolación depende de la forma y estabilidad de la curva observada.
- Los resultados pueden variar según el conjunto de datos y la configuración de los hiperparámetros.
- El porcentaje de curva utilizado para realizar el ajuste puede influir en la calidad de las predicciones.
- Un error de predicción puede provocar que un ensayo se interrumpa antes de alcanzar un rendimiento potencialmente mejor.
- El ahorro computacional no es necesariamente superior al de la parada anticipada convencional en todos los escenarios.

Por este motivo, resulta importante evaluar el método en diferentes conjuntos de datos y configuraciones experimentales.

---

## Autoría

Proyecto desarrollado como parte de un Trabajo de Fin de Grado centrado en el estudio de criterios de parada anticipada basados en la extrapolación de curvas de aprendizaje dentro de la optimización bayesiana de hiperparámetros.