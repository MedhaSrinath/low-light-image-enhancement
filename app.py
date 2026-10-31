
import io
from pathlib import Path

import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageOps

from model import EnhancementCNN


st.set_page_config(
    page_title="Low-Light Image Enhancement",
    page_icon="🌙",
    layout="wide"
)

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
IMAGE_SIZE = (384, 256)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f0ff, #fff7fb, #f0f4ff);
    color: #403453;
}

.block-container {
    max-width: 1050px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #514071 !important;
}

p, label {
    color: #655a78;
}

.header {
    background: linear-gradient(120deg, #e5d9ff, #f8e2f0);
    padding: 38px 25px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 35px;
}

.header h1 {
    font-size: 34px;
    margin-bottom: 12px;
}

.header p {
    font-size: 16px;
    margin-bottom: 0;
}

.section-heading {
    font-size: 23px;
    font-weight: 650;
    color: #594477;
    margin: 25px 0 10px 0;
}

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #d9cbed;
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stFileUploader"] section {
    background: #faf8ff !important;
    border: 1px dashed #bca9df !important;
    border-radius: 12px !important;
}

div[data-testid="stFileUploader"] button {
    background: #e9e0f8 !important;
    color: #594477 !important;
    border: 1px solid #cbb9e7 !important;
    border-radius: 8px !important;
}

div.stButton > button {
    background: #8064b5;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background: #6c519f;
    color: white;
    border: none;
}

div[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
}

.image-label {
    background: white;
    border: 1px solid #e5dcf0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    color: #594477;
    font-weight: 600;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #9385a5;
    font-size: 13px;
    margin-top: 45px;
    padding: 15px;
    border-top: 1px solid #e7dff0;
}

.stDownloadButton button {
    background: #8064b5 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Trained model checkpoint not found. "
            "Please add checkpoints/best_model.pt."
        )

    model = EnhancementCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def enhance_image(image, model, device):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size

    resized_image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    ) / 255.0

    input_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor).clamp(0, 1)

    output_array = (
        output[0].cpu().permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)

    enhanced_image = Image.fromarray(output_array)

    return enhanced_image.resize(
        original_size,
        Image.Resampling.BILINEAR
    )


# Header
st.markdown("""
<div class="header">
    <h1>Low-Light Image Enhancement</h1>
    <p>Improve the visibility of dark images using a CNN-based deep learning model.</p>
</div>
""", unsafe_allow_html=True)


# Upload
st.markdown(
    '<div class="section-heading">Upload Image</div>',
    unsafe_allow_html=True
)

st.write("Select a low-light image from your device.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"]
)


if uploaded_file is not None:

    original_image = Image.open(uploaded_file)
    original_image = ImageOps.exif_transpose(
        original_image
    ).convert("RGB")

    st.markdown(
        '<div class="section-heading">Selected Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        original_image,
        use_container_width=True
    )

    if st.button("Enhance Image"):

        try:
            with st.spinner("Processing image..."):

                model, device = load_model()

                enhanced_image = enhance_image(
                    original_image,
                    model,
                    device
                )

            st.success("Image enhancement completed!")

            st.markdown(
                '<div class="section-heading">Before and After Comparison</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    '<div class="image-label">Original Image</div>',
                    unsafe_allow_html=True
                )
                st.image(
                    original_image,
                    use_container_width=True
                )

            with col2:
                st.markdown(
                    '<div class="image-label">Enhanced Image</div>',
                    unsafe_allow_html=True
                )
                st.image(
                    enhanced_image,
                    use_container_width=True
                )

            image_buffer = io.BytesIO()
            enhanced_image.save(image_buffer, format="PNG")

            st.download_button(
                "Download Enhanced Image",
                data=image_buffer.getvalue(),
                file_name="enhanced_image.png",
                mime="image/png"
            )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Something went wrong: {error}")

else:
    st.info("Upload an image to get started.")


st.markdown("""
<div class="footer">
    CNN-Based Low-Light Image Enhancement Project
</div>
""", unsafe_allow_html=True)
