import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

# Set Streamlit page configuration to wide layout
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .st-emotion-cache-1jicfl2 {
width: 100%;
padding: 6rem 1rem 10rem;
min-width: auto;
max-width: initial;
}
    """,
    unsafe_allow_html=True,
)


def normalize_image(image, vmin, vmax):
    """Normalize the image to the range [0, 1] based on vmin and vmax."""
    normalized = np.clip(image, vmin, vmax)
    normalized = (normalized - vmin) / (vmax - vmin)
    return normalized

def main():
    # Set title for the app
    st.title("Image Comparison")

    # Adjust column widths and add a spacer column (e.g., 1:0.1:3)
    col1, spacer, col2 = st.columns([2, 0.8, 5])  # Adjust the middle number to set the space size

    with col1:
        # File uploaders in the left column
        uncalibrated_image = st.file_uploader("Choose uncalibrated image", type="tiff")
        calibrated_image = st.file_uploader("Choose calibrated image", type="tiff")

    with col2:
        if calibrated_image and uncalibrated_image:
            # Open images
            img1 = Image.open(calibrated_image)
            img2 = Image.open(uncalibrated_image)

            # Convert to numpy arrays and ensure they're float type
            img1_array = np.array(img1).astype(float)
            img2_array = np.array(img2).astype(float)

            # Check if images have the same dimensions
            if img1_array.shape != img2_array.shape:
                st.error("Images must have the same dimensions")
                return

            # Normalize images based on vmin and vmax
            img1_norm = normalize_image(img1_array, 0.001, 0.1)
            img2_norm = normalize_image(img2_array, 0.001, 0.1)

            # Use the image_comparison component in the right column
            image_comparison(
                img1=img1_norm,
                img2=img2_norm,
                label1="Calibrated",
                label2="Uncalibrated",
                width=700,  # Adjust the width of the comparison slider
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

if __name__ == "__main__":
    main()
