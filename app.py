import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detector PPE", page_icon="🦺", layout="wide")

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def infer_image(image, model, conf):
    # Inferencia
    results = model(image, conf=conf)
    result = results[0]

    # Imagen anotada
    annotated = result.plot()
    annotated = annotated[:, :, ::-1]  # BGR -> RGB

    # Tabla de detecciones
    rows = []
    names = result.names

    if result.boxes is not None and len(result.boxes) > 0:
        classes = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist()

        for cls_id, score in zip(classes, confs):
            rows.append({
                "clase": names[int(cls_id)],
                "confianza": round(float(score), 4)
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["clase", "confianza"])
    return annotated, df

st.title("🦺 Reconocimiento de PPE")
st.write("Sube una imagen o toma una foto para detectar elementos de protección personal.")

with st.sidebar:
    st.header("Configuración")
    conf = st.slider("Confianza mínima", 0.05, 0.95, 0.25, 0.05)
    source = st.radio("Fuente", ["Subir imagen", "Tomar foto"])

try:
    model = load_model()
except Exception as e:
    st.error(f"No se pudo cargar el modelo '{MODEL_PATH}': {e}")
    st.stop()

image = None

if source == "Subir imagen":
    uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
else:
    camera_file = st.camera_input("Toma una foto")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen original")
        st.image(image, use_container_width=True)

    with st.spinner("Ejecutando detección..."):
        annotated, df = infer_image(image, model, conf)

    with col2:
        st.subheader("Resultado")
        st.image(annotated, use_container_width=True)

    st.subheader("Detecciones")
    if df.empty:
        st.info("No se detectaron elementos PPE.")
    else:
        st.dataframe(df, use_container_width=True)

        resumen = df["clase"].value_counts().reset_index()
        resumen.columns = ["clase", "cantidad"]

        st.subheader("Resumen por clase")
        st.dataframe(resumen, use_container_width=True)
else:
    st.info("Esperando imagen...")