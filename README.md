# 🟡 Predicción de Recuperación de Oro en Planta Minera

[![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.10-blue.svg)](https://www.python.org/)
[![Data Science Bootcamp](https://img.shields.io/badge/Proyecto-Bootcamp-green)](#)
[![Status](https://img.shields.io/badge/Status-Completo-brightgreen)](#)

---

Este proyecto pertenece al sector **minero** y está enfocado en modelar la **recuperación de oro** en una planta de procesamiento.  

Se construyen modelos de **Machine Learning** para predecir la eficiencia de recuperación en dos etapas clave del proceso:

- `rougher.output.recovery`
- `final.output.recovery`

El desempeño se evalúa mediante la métrica **sMAPE (Symmetric Mean Absolute Percentage Error)**, tanto por etapa como en una métrica combinada ponderada.

---

## 📌 Objetivo

Desarrollar un modelo capaz de:

- Predecir la recuperación de oro en las etapas **rougher** y **final**.
- Comparar el desempeño de **Regresión Lineal** y **Random Forest**.
- Evaluar la calidad del modelo usando una métrica específica del negocio (**sMAPE**).
- Obtener un indicador final de la calidad del sistema de predicción combinando ambas etapas.

---

## 🛠️ Herramientas utilizadas

- `Python`
- `Pandas`, `NumPy`
- `scikit-learn` (LinearRegression, RandomForestRegressor, KFold, Pipeline)
- `Matplotlib`, `Seaborn` (para EDA)
- `SciPy` (pruebas estadísticas puntuales)
- `Jupyter Notebook`

---

## 📊 Contenido del análisis

- ✔ Carga y exploración de los datasets:
  - `gold_recovery_train.csv`
  - `gold_recovery_test.csv`
  - `gold_recovery_full.csv`
- ✔ Análisis de:
  - Estructura de columnas y tipos de datos.
  - Composición y pureza de concentrados en las distintas etapas.
  - Distribuciones de variables clave (Au, Ag, Pb).
- ✔ Preparación de datos:
  - Selección de **features** que existen tanto en train como en test.
  - Exclusión de columnas de salida (`rougher.output.*`, `final.output.*`) como predictores.
  - Revisión y tratamiento básico de valores ausentes.
- ✔ Definición de la métrica de negocio:
  - Implementación de `sMAPE` en porcentaje.
  - Métrica final ponderada:
    - 25% `rougher.output.recovery`
    - 75% `final.output.recovery`
- ✔ Modelado:
  - Creación de un **pipeline** (`SimpleImputer` + `StandardScaler` + modelo).
  - Entrenamiento y evaluación con **validación cruzada (KFold)**.
  - Comparación de:
    - **Regresión Lineal**
    - **Random Forest Regressor**
- ✔ Selección del mejor modelo según sMAPE combinado.

---

## 📈 Resultados clave

- La **Regresión Lineal** logra resultados razonables, pero es sensible a outliers y relaciones no lineales presentes en el proceso metalúrgico.
- El **Random Forest Regressor**:
  - Mejora el error sMAPE tanto en la etapa *rougher* como en la *final*.
  - Presenta mejor capacidad para capturar relaciones complejas entre las variables del proceso.
- En términos de métrica combinada (`sMAPE_rougher`, `sMAPE_final`, `sMAPE_combinado`), el modelo de **Random Forest** obtiene el menor `sMAPE_combinado`, por lo que se selecciona como modelo final.

> Ejemplo de tabla de resultados (valores ilustrativos):

| modelo | sMAPE_rougher | sMAPE_final | sMAPE_combinado |
|-------|----------------|-------------|------------------|
| rf    | 7.68           | 6.72        | 6.96            |
| linreg| 10.16          | 9.09        | 9.36            |

---

## 🧠 Conclusión

- El uso de **Random Forest** permite construir un modelo más robusto para predecir la recuperación de oro en planta.
- La métrica **sMAPE combinada** refleja mejor el impacto global del sistema, dando mayor peso a la recuperación final.
- Este enfoque:
  - Ayuda a entender el desempeño del proceso metalúrgico.
  - Puede apoyar decisiones de operación y optimización en el sector minero.
- Este proyecto se integra a mi portafolio como un ejemplo claro de **Ciencia de Datos aplicada a la minería**, conectando experiencia industrial con herramientas de **Machine Learning**.

---

## 📁 Estructura del proyecto

```text
Recuperacion-Oro-Mineria/

├── Src/
│   └── Modelo_Recuperacion_Oro.py          # Código fuente limpio con el pipeline de ML
│
├── Notebooks/
│   └── Sprint12_ProyectoZyfra.ipynb        # Notebook con el desarrollo paso a paso
│
├── Data/
│   ├── gold_recovery_train.csv
│   ├── gold_recovery_test.csv
│   └── gold_recovery_full.csv
│
│   Nota: > Nota: El archivo `gold_recovery_full.csv` no se incluye en el repositorio debido a su tamaño.  
│         > Para reproducir el experimento completo, consulta las instrucciones en `Data/README_DATA.md`.
│
├── requirements.txt                        # Librerías necesarias
├── .gitignore
└── README.md
```
## 👨‍💻 Autor

Axel López


🔗 LinkedIn - https://www.linkedin.com/in/axel-lópez-linares/

✉️ axellpzlin@gmail.com

🎯 Proyecto de portafolio - Bootcamp de Ciencia de Datos (Oro / Minería)
