import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo previamente entrenado
print("Cargando modelo...")
modelo = tf.keras.models.load_model("modelo_letras_mlp.h5")
letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# a. Captura de cámara [cite: 68]
cap = cv2.VideoCapture(0)

print("Iniciando cámara... Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Voltear el frame como un espejo para que sea más intuitivo
    frame = cv2.flip(frame, 1)
    
    # b. Selección de una región de interés [cite: 69]
    # Coordenadas del recuadro verde en el centro
    x_start, y_start, width, height = 200, 150, 200, 200
    cv2.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), (0, 255, 0), 2)
    
    # Extraer la imagen dentro del recuadro
    roi = frame[y_start:y_start+height, x_start:x_start+width]
    
    # c. Preprocesamiento de la imagen [cite: 70, 71]
    # Convertir a escala de grises
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir el ruido del fondo
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Umbralización adaptativa para resaltar la letra negra sobre fondo blanco (simulando papel)
    _, binaria = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # d. Redimensionamiento a 50x50 píxeles [cite: 72]
    roi_resized = cv2.resize(binaria, (50, 50))
    
    # Preparar los datos para la red neuronal (Normalización)
    entrada_modelo = roi_resized.astype('float32') / 255.0
    entrada_modelo = entrada_modelo.reshape(1, 50, 50, 1)
    
    # e. Predicción con el modelo entrenado [cite: 73]
    prediccion = modelo.predict(entrada_modelo, verbose=0)
    clase_predicha = np.argmax(prediccion)
    letra_predicha = letras[clase_predicha]
    probabilidad = np.max(prediccion) * 100
    
    # f. Visualización de la letra predicha [cite: 74]
    # Solo mostrar la predicción si hay cierta confianza para evitar parpadeos con ruido
    if probabilidad > 60:
        texto = f"Letra: {letra_predicha} ({probabilidad:.1f}%)"
        cv2.putText(frame, texto, (x_start, y_start - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Coloca una letra...", (x_start, y_start - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    
    # Mostrar ventanas en pantalla
    cv2.imshow('1. Captura en tiempo real', frame)
    cv2.imshow('2. Imagen procesada para el modelo', roi_resized)
    
    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()