import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Carregar o modelo
model = YOLO('best_license_plate_model.pt')

def process_frame(frame):
    # Converter o frame para formato que o OpenCV pode manipular
    image_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar a predição no frame
    results = model.predict(image_cv, device='cpu')

    detected_texts = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            roi = frame[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, config='--psm 6')
            detected_texts.append((text, confidence))

    return frame, detected_texts

# Interface do Streamlit
st.title("Detecção de Placas de Carro em Vídeo ao Vivo")

# Inicializar a webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Erro ao acessar a webcam")
else:
    stframe = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Erro ao capturar o frame")
            break

        # Processar o frame
        processed_frame, detected_texts = process_frame(frame)

        # Converter o frame para RGB para exibição no Streamlit
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Exibir o frame no Streamlit
        stframe.image(processed_frame_rgb, use_column_width=True)

        # Exibir textos detectados
        st.write("Textos Detectados:")
        for text, confidence in detected_texts:
            st.write(f"**Texto:** {text}")
            st.write(f"**Confiança:** {confidence*100:.2f}%")

    cap.release()
    cv2.destroyAllWindows()
