# Clasificador K-NN sobre agrupaciones de K-Means

Aplicación web del Taller 6 (Herramientas de Inteligencia Artificial, ICYA3004).
Clasifica nuevas observaciones en las agrupaciones descubiertas por K-Means, usando
un clasificador K-NN entrenado sobre esas etiquetas.

## Contenido

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit con dos pestañas |
| `requirements.txt` | Dependencias (`scikit-learn` fijado a 1.8.0) |
| `runtime.txt` | Versión de Python |
| `modelos/knn_iris.joblib` | K-NN del caso canónico Iris (k=5) |
| `modelos/scaler_iris.joblib` | StandardScaler de Iris |
| `modelos/metadata_iris.json` | Variables, rangos y centroides de Iris |
| `modelos/knn_model.joblib` | K-NN de los envíos de e-commerce (k=5) |
| `modelos/scaler.joblib` | StandardScaler de los envíos |
| `modelos/metadata.json` | Variables, rangos y perfiles de los segmentos |

## Modelos

**Iris Plant** (caso canónico): K-Means con k=3 (silueta 0.4599, ARI 0.6201 frente a
las especies reales) y K-NN con k=5 vecinos, 95.6% de exactitud.

**Envíos de e-commerce**: K-Means con k=4 (silueta 0.3619) y K-NN con k=5 vecinos,
98.4% de exactitud. El segmento 0 concentra las promociones agresivas sobre paquetes
livianos y presenta 99.6% de tasa histórica de retraso.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota

Las agrupaciones provienen de K-Means, no de etiquetas reales. En Iris coinciden
exactamente con *setosa*, pero *versicolor* y *virginica* se solapan, por lo que la
frontera entre esos dos grupos es geométrica y no biológica. En los envíos, el
porcentaje mostrado es la tasa histórica del segmento y no una predicción individual.
