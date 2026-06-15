"""
====================================================================
=  PROYECTO FINAL                                                  =
=  DESARROLLO DE UNA LIBRERIA EN PYTHON PARA LA DETECCION DE       =
=  LETRAS CON UNA CAMARA                                           =
=                                                                  =
=    = 
=           =
=   =
=     =
= = 
=   =
=                                                      =
====================================================================
  INTEGRANTES:
  - HAN APESS ESPARZA
  - ARMANDO DUARTE ESPARZA
  - IKER EDUARDO FIGUEROA GARCIA
  - JOSE IVAN RODRIGUEZ VALTIERRA                                                  
"""

import numpy as np
import glob
import os

#Buscar el archivo de los datos obtenidos (.npy)
Ruta_Archivos = "Datos_Individual"

#Sacar y buscar todo los archivos que empiecen con X Y .npy
Archivos_X = sorted(glob.glob(os.path.join(Ruta_Archivos, "X*.npy")))
Archivos_Y = sorted(glob.glob(os.path.join(Ruta_Archivos, "y*.npy")))

#Por si encuentran los archivos
print (f"Se encontraron{len(Archivos_X)} archivos (X) y {len(Archivos_Y)} archivos (Y)" )

#Colocamos listas para sacar los datos 
Lista_X = []
Lista_Y = []

#Cargar y aplicar todos los datasets
for fx, fy in zip (Archivos_X, Archivos_Y):
    Lista_X.append(np.load(fx, allow_pickle=True))
    Lista_Y.append(np.load(fy, allow_pickle=True))

#Se une todo en un solo arreglo para hacer mas sencillo el uso de todos los archivos
X_Completo = np.concatenate(Lista_X, axis=0)
Y_Completo = np.concatenate(Lista_Y, axis=0)

#Todos estos archivos individuales deberian integrarse en un solo dataset
np.save("X_Completo.npy", X_Completo)
np.save("Y_Completo.npy", Y_Completo)

print("Dataset integrado y guardado exitosamente")
print(f"Forma final de X_grupo: {X_Completo.shape}")
print(f"Forma final de y_grupo: {Y_Completo.shape}")

