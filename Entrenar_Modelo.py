"""
====================================================================
=  PROYECTO FINAL                                                  =
=  ENTRENAMIENTO DEL MODELO MLP PARA DETECCIÓN DE LETRAS           =
====================================================================
  INTEGRANTES:
  - HAN APESS ESPARZA
  - ARMANDO DUARTE ESPARZA
  - IKER EDUARDO FIGUEROA GARCIA
  - JOSE IVAN RODRIGUEZ VALTIERRA                                  
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar el dataset grupal consolidado
print("Cargando los datos...")
X_Completo = np.load("X_Completo.npy", allow_pickle=True)
Y_Completo = np.load("Y_Completo.npy", allow_pickle=True)

# 2. Preparación de los datos para la red neuronal
# Los valores de intensidad deberán normalizarse.
X_Completo = X_Completo.astype('float32') / 255.0

# Procesar las etiquetas dinámicamente con LabelEncoder y convertirlas a formato one-hot.
le = LabelEncoder()
Y_encoded = le.fit_transform(Y_Completo)
Y_onehot = to_categorical(Y_encoded, num_classes=26)

print(f"Se detectaron y codificaron {len(le.classes_)} clases distintas.")

# 3. División del dataset
# El dataset grupal deberá dividirse en conjuntos de entrenamiento (80%) y prueba (20%).
# La división deberá realizarse de forma aleatoria y estratificada.
X_train, X_test, y_train, y_test = train_test_split(
    X_Completo, Y_onehot, 
    test_size=0.20, 
    random_state=42, 
    stratify=Y_onehot 
)

y_test_original = np.argmax(y_test, axis=1)

# Asegurar que la forma final de los datos de entrada cuadre con los 50x50 píxeles.
X_train = X_train.reshape((X_train.shape[0], 50, 50, 1))
X_test = X_test.reshape((X_test.shape[0], 50, 50, 1))

# 4. Modelo de red neuronal densa (MLP)
learning_rate = 0.001
modelo = Sequential()

# La capa de entrada recibe los 2500 valores de cada imagen.
modelo.add(Flatten(input_shape=(50, 50, 1))) 

# Las capas ocultas aprenden patrones generales de la forma de las letras.
modelo.add(Dense(512, activation='relu'))
modelo.add(Dense(256, activation='relu'))
modelo.add(Dense(128, activation='relu'))

# La capa de salida tendrá 26 neuronas, una por cada letra mayúscula del alfabeto.
modelo.add(Dense(26, activation='softmax')) 

optimizador = tf.keras.optimizers.Adam(learning_rate=learning_rate)

modelo.compile(
    optimizer=optimizador,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# 5. Entrenamiento del modelo
print("Iniciando entrenamiento...")
historial = modelo.fit(
    X_train, y_train,
    epochs=15,          
    batch_size=128,
    validation_split=0.2 
)

# 6. Evaluación del modelo
test_loss, test_accuracy = modelo.evaluate(X_test, y_test)

# Reporte de exactitud global.
print(f"\nExactitud global del modelo en prueba: {test_accuracy:.4f}") 

# Gráfica de exactitud y pérdida durante el entrenamiento.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Exactitud del Modelo')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()

# Matriz de confusión.
predicciones = modelo.predict(X_test)
predicciones_clases = np.argmax(predicciones, axis=1)

matriz_conf = confusion_matrix(y_test_original, predicciones_clases)
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Ejemplos visuales de predicciones correctas y errores de clasificación.
plt.figure(figsize=(12, 6))
indices_aleatorios = np.random.choice(len(X_test), 15, replace=False)

for i, idx in enumerate(indices_aleatorios):
    plt.subplot(3, 5, i + 1)
    imagen = X_test[idx].reshape(50, 50)
    plt.imshow(imagen, cmap='gray')
    
    # Decodificar de números (0-25) a las letras originales ('A'-'Z')
    etiqueta_real = le.inverse_transform([y_test_original[idx]])[0]
    prediccion = le.inverse_transform([predicciones_clases[idx]])[0]
    
    color = 'green' if etiqueta_real == prediccion else 'red'
    plt.title(f"Real: {etiqueta_real}\nPred: {prediccion}", color=color)
    plt.axis('off')

plt.suptitle("Ejemplos visuales de predicciones")
plt.tight_layout()
plt.show()

# Guardar modelo
modelo.save("modelo_letras_mlp.h5")
print("\n¡Modelo entrenado y guardado exitosamente como modelo_letras_mlp.h5!")