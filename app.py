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


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("Trained model checkpoint not found.")

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


st.title("🌙 Low-Light Image Enhancement")

st.write(
    "Enhance dark and low-light images using "
    "a Convolutional Neural Network (CNN)."
)

st.divider()

st.subheader("Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose a dark image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file)
    original_image = ImageOps.exif_transpose(
        original_image
    ).convert("RGB")

    if st.button("✨ Enhance Image", type="primary"):

        try:
            with st.spinner("Enhancing your image..."):

                model, device = load_model()

                enhanced_image = enhance_image(
                    original_image,
                    model,
                    device
                )

            st.success("Image enhancement completed!")

            st.subheader("Before and After Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(
                    original_image,
                    use_container_width=True
                )

            with col2:
                st.markdown("### Enhanced Image")
                st.image(
                    enhanced_image,
                    use_container_width=True
                )

            image_buffer = io.BytesIO()
            enhanced_image.save(image_buffer, format="PNG")

            st.download_button(
                "⬇️ Download Enhanced Image",
                data=image_buffer.getvalue(),
                file_name="enhanced_image.png",
                mime="image/png"
            )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Error: {error}")

else:
    st.info("Upload an image to get started.")
