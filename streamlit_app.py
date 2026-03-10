import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import gdown
import os
import time
import numpy as np

# =========================
# Download model nếu chưa có
# =========================
url = "https://drive.google.com/uc?id=1xZL5HjrVFsp4Zu6osnc_btM1NfiBcoop"
model_path = "best.pt"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# =========================
# Load model (chỉ load 1 lần)
# =========================
@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

# =========================
# Predict function
# =========================
def predict(image):

    start = time.time()

    results = model.predict(image)[0]
    image_pred = results.plot()

    # tạo thư mục lưu kết quả
    os.makedirs("results", exist_ok=True)

    # tạo tên file theo timestamp
    filename = f"results/pred_{int(time.time())}.png"

    cv2.imwrite(filename, image_pred)

    end = time.time()
    pred_time = round(end - start, 2)

    return image_pred, filename, pred_time


# =========================
# Main UI
# =========================
def main():

    st.title("Human Vasculature Image Segmentation")

    # =========================
    # Test images
    # =========================
    val_path = "val"

    if os.path.exists(val_path):

        image_files = os.listdir(val_path)

        if image_files:

            st.header("Select a test image")

            selected_image = st.selectbox("Test images", image_files)

            image_path = os.path.join(val_path, selected_image)

            image = Image.open(image_path)

            st.image(image, caption="Original image", use_column_width=True)

            if st.button("Predict test image"):

                with st.spinner("Running prediction..."):

                    pred_img, save_path, pred_time = predict(image)

                st.image(pred_img, caption="Prediction result", use_column_width=True)

                st.success(f"Prediction time: {pred_time} seconds")

                with open(save_path, "rb") as file:
                    st.download_button(
                        label="Download result",
                        data=file,
                        file_name="prediction.png",
                        mime="image/png"
                    )

    # =========================
    # Upload image
    # =========================
    st.header("Upload an image")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "tif"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Predict uploaded image"):

            with st.spinner("Running prediction..."):

                pred_img, save_path, pred_time = predict(image)

            st.image(pred_img, caption="Prediction result", use_column_width=True)

            st.success(f"Prediction time: {pred_time} seconds")

            with open(save_path, "rb") as file:
                st.download_button(
                    label="Download result",
                    data=file,
                    file_name="prediction.png",
                    mime="image/png"
                )


# =========================
# Run app
# =========================
if __name__ == "__main__":
    main()