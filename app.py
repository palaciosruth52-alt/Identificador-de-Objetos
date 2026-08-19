import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="ImageClassify",
    page_icon="",
    layout="wide"
)

MODEL_PATH = "modelo_cifar10.keras"
CLASSES_PATH = "class_names.json"

IMAGE_SIZE = (32, 32)


CLASS_INFO = {
    "Avión": "✈️",
    "Automóvil": "🚗",
    "Pájaro": "🐦",
    "Gato": "🐱",
    "Ciervo": "🦌",
    "Perro": "🐕",
    "Rana": "🐸",
    "Caballo": "🐎",
    "Barco": "🚢",
    "Camión": "🚚"
}


@st.cache_resource
def cargar_modelo():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "No se encontró el archivo modelo_cifar10.keras"
        )

        st.stop()

    modelo = tf.keras.models.load_model(
        MODEL_PATH
    )

    return modelo

@st.cache_data
def cargar_clases():

    if not os.path.exists(CLASSES_PATH):

        st.error(
            "No se encontró el archivo class_names.json"
        )

        st.stop()

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as archivo:

        clases = json.load(archivo)

    return clases

modelo = cargar_modelo()

class_names = cargar_clases()

st.markdown(
    """
    <div style="text-align:center">

    <h1>🖼️ ImageClassify</h1>

    <h3>
    Clasificación de imágenes mediante Machine Learning
    </h3>

    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    """
    Esta aplicación utiliza una Red Neuronal Convolucional
    entrenada con el dataset CIFAR-10 para identificar
    diferentes categorías de objetos a partir de imágenes.
    """
)

st.sidebar.title("👤 Información")

st.sidebar.write(
    "**Autor:** Ruth Palacios"
)

st.sidebar.write(
    "**Modelo:** CNN"
)

st.sidebar.write(
    "**Dataset:** CIFAR-10"
)

st.sidebar.write(
    "**Categorías:** 10"
)

st.header(
    "📷 Seleccionar imagen"
)

st.write(
    "Puedes cargar una imagen existente o tomar una fotografía."
)


col1, col2 = st.columns(2)

with col1:

    archivo = st.file_uploader(
        " Subir una imagen",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

with col2:

    foto = st.camera_input(
        " Tomar una fotografía"
    )

imagen_archivo = None

if archivo is not None:

    imagen_archivo = archivo

elif foto is not None:

    imagen_archivo = foto

if imagen_archivo is not None:

    imagen = Image.open(
        imagen_archivo
    ).convert("RGB")


    st.divider()

    col_imagen, col_resultado = st.columns(
        [1, 1]
    )

    with col_imagen:

        st.subheader(
            "🖼️ Imagen seleccionada"
        )

        st.image(
            imagen,
            caption="Imagen proporcionada por el usuario",
            use_container_width=True
        )

    imagen_procesada = imagen.resize(
        IMAGE_SIZE
    )

    imagen_array = np.array(
        imagen_procesada
    )

    imagen_array = (
        imagen_array.astype("float32")
        / 255.0
    )

    imagen_array = np.expand_dims(
        imagen_array,
        axis=0
    )

    with st.spinner(
        "🤖 Analizando la imagen..."
    ):

        predicciones = modelo.predict(
            imagen_array,
            verbose=0
        )[0]


    indice = int(
        np.argmax(predicciones)
    )

    clase = class_names[indice]

    confianza = float(
        predicciones[indice]
    )

    with col_resultado:

        st.subheader(
            " Resultado"
        )

        icono = CLASS_INFO.get(
            clase,
            "🔎"
        )

        st.success(
            f"{icono} Objeto identificado: {clase}"
        )

        st.metric(
            "Porcentaje de confianza",
            f"{confianza * 100:.2f}%"
        )

        st.progress(
            confianza
        )


    st.divider()

    st.subheader(
        "📊 Probabilidad por categoría"
    )

    for i, nombre in enumerate(class_names):

        probabilidad = float(
            predicciones[i]
        )

        icono = CLASS_INFO.get(
            nombre,
            "🔎"
        )

        st.write(
            f"{icono} **{nombre}** — "
            f"{probabilidad * 100:.2f}%"
        )

        st.progress(
            probabilidad
        )


