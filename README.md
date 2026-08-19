## Servicio Web de Clasificación de Imágenes mediante Machine Learning

ImageClassify es una aplicación web desarrollada con Python y Streamlit que utiliza un modelo de aprendizaje automático para identificar y clasificar objetos presentes en imágenes.

El sistema permite al usuario cargar una imagen desde su dispositivo o tomar una fotografía utilizando la cámara. Posteriormente, la imagen es procesada por una Red Neuronal Convolucional (CNN), que genera una predicción y muestra la categoría identificada junto con su porcentaje de confianza.

El proyecto fue desarrollado utilizando Google Colab para el entrenamiento del modelo y Streamlit Community Cloud para el despliegue de la aplicación.

#  Objetivo

Desarrollar e implementar un Servicio Web basado en Machine Learning capaz de clasificar imágenes utilizando un modelo de Redes Neuronales Convolucionales y ponerlo a disposición de los usuarios mediante una aplicación web desplegada en la nube.

#  Funcionalidades

La aplicación permite:

-  Subir imágenes desde el dispositivo.
-  Tomar fotografías utilizando la cámara.
-  Mostrar la imagen seleccionada.
-  Clasificar la imagen mediante Machine Learning.
-  Mostrar el porcentaje de confianza.
-  Mostrar las probabilidades de todas las categorías.
-  Ejecutarse como una aplicación web en la nube.
-  Mostrar información del autor.

#  Dataset utilizado

El proyecto utiliza el dataset **CIFAR-10**.

CIFAR-10 contiene imágenes clasificadas en diez categorías:

| Número | Categoría |
|---:|---|
| 0 |  Avión |
| 1 |  Automóvil |
| 2 |  Pájaro |
| 3 |  Gato |
| 4 |  Ciervo |
| 5 |  Perro |
| 6 |  Rana |
| 7 |  Caballo |
| 8 |  Barco |
| 9 |  Camión |
| 10 | Persona |

Las imágenes originales del dataset tienen un tamaño de 32 × 32 píxeles y tres canales de color RGB.


#  Modelo de Machine Learning

Para realizar la clasificación se desarrolló una **Red Neuronal Convolucional (CNN)** utilizando TensorFlow y Keras.

La arquitectura utiliza:

- Capa de entrada.
- Capas Conv2D.
- Capas MaxPooling2D.
- Capa Flatten.
- Capa Dense.
- Capa de salida con 10 categorías.

La capa final utiliza la función de activación Softmax para obtener las probabilidades correspondientes a cada categoría.

#  Resultados del modelo

El modelo fue evaluado utilizando el conjunto de prueba de CIFAR-10.

### Resultado general

**Accuracy: 71.26%**

El modelo obtuvo los siguientes resultados:

| Clase | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Avión | 72% | 76% | 74% |
| Automóvil | 78% | 87% | 82% |
| Pájaro | 63% | 59% | 61% |
| Gato | 52% | 56% | 54% |
| Ciervo | 66% | 67% | 67% |
| Perro | 67% | 51% | 58% |
| Rana | 88% | 70% | 78% |
| Caballo | 69% | 83% | 75% |
| Barco | 79% | 83% | 81% |
| Camión | 80% | 81% | 80% |


El resultado muestra que el modelo presenta un mejor desempeño en categorías como Automóvil, Barco, Camión y Rana.

Las categorías Gato, Perro y Pájaro presentan mayor dificultad de clasificación debido a la similitud visual entre algunas imágenes.

---

#  Arquitectura del sistema

La arquitectura general del proyecto es:

```text
                    USUARIO
                       │
                       ▼
              ┌─────────────────┐
              │    Streamlit    │
              │   Aplicación    │
              │      Web        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Imagen cargada  │
              │ o fotografía    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Preprocesamiento│
              │     32 × 32     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │      CNN        │
              │ TensorFlow/Keras│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Predicción    │
              │  + Confianza    │
              └────────┬────────┘
                       │
                       ▼
                    USUARIO
