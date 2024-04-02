import streamlit as st

from PIL import Image, ImageFilter, ImageOps
from transformers import pipeline

# model = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")

@st.cache_resource()
def load_model_pipeline(task, model_path):
    model = pipeline(task, model=model_path)
    return model

#### Image Generation ####
st.title("Background Blur")
model_path = ("./Models/models--mattmdjaga--segformer_b2_clothes/"
              "snapshots/f6ac72992f938a1d0073fb5e5a06fd781f19f9a2")

model = load_model_pipeline('image-segmentation', model_path)

# File uploader with the unique key from session state
uploaded_image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_image:
    with st.spinner('Adding Blur...'):
        blur_level = st.slider("Adjust Blur Level", min_value=0, max_value=30, value=15, step=1)

        original = Image.open(uploaded_image)
        result = model(images=original)

        # Background
        mask = result[0]['mask']
        # st.image(mask, caption="Background")

        mask_original = Image.composite(original, Image.new('RGB', original.size, 0), mask)

        mask_original_blur = mask_original.filter(ImageFilter.GaussianBlur(radius=blur_level))
        # st.image(mask_original_blur, caption="mask_original_blur")

        mask_inverted = ImageOps.invert(mask)
        # st.image(mask_inverted, caption="mask_inverted")

        mask_inverted_original = Image.composite(original,
                                                 Image.new('RGB', original.size, 0), mask_inverted)
        # st.image(mask_inverted_original, caption="mask_inverted_original")

        final_image = Image.composite(mask_inverted_original, mask_original_blur, mask_inverted)
        st.image(final_image, caption="final_image")