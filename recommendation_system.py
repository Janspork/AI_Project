# recommendation_system.py

# Importar las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Cargar los datos (ejemplo: dataset de productos)
# Nota: Este dataset es un ejemplo simulado basado en características
# En un caso real, leeríamos un archivo: data = pd.read_csv("products.csv")

# Simulamos unos datos rápidos para que el script funcione perfectamente si lo ejecutas:
np.random.seed(42)
X_mock = np.random.rand(100, 3)  # 100 productos, 3 características cada uno
y_mock = np.random.choice([0, 1], size=100)  # Categorías de recomendación (0 o 1)

# Procesamiento de datos
# Seleccionar las características y la etiqueta
features = X_mock  # En la guía: data[["feature1", "feature2", "feature3"]]
labels = y_mock    # En la guía: data["label"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Función de recomendación
def recommend(product_features):
    prediction = model.predict(product_features)
    return prediction

# Ejemplo de uso de la función de recomendación
example_product = np.array([[0.5, 0.2, 0.8]])  # Características del producto a evaluar
recommended_product = recommend(example_product)
print(f"Recommended Product Class: {recommended_product}")