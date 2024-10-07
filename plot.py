import streamlit as st
from PIL import Image
import numpy as np
from streamlit_image_comparison import image_comparison

# Set Streamlit page configuration to wide layout and custom page height
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Label styles */
    .image-comparison-label {
        background-color: rgba(0, 0, 255, 0.8); /* Semi-transparent white */
        padding: 5px 10px;
        border-radius: 3px;
        position: absolute; /* Make it positionable */
        z-index: 10; /* Ensure it's above the images */
    }
    
    /* Position for left and right labels */
    .label1 {
        left: 10px; /* Position for the left label */
        top: 5px; /* Adjust top position as needed */
    }
    
    .label2 {
        right: 10px; /* Position for the right label */
        top: 5px; /* Adjust top position as needed */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Create a container for better layout control
    with st.container():
        # Split the page into two columns with custom ratios
        col1, _, col2 = st.columns([2.5, 1, 5])

        with col1:
            st.markdown("<h2 class='subheader'>Image Comparison Tool</h2>", unsafe_allow_html=True)
            
            # File uploaders with custom styling
            uncalibrated_image = st.file_uploader(
                "Upload uncalibrated image",
                type="png",
                help="Select a PNG file for the uncalibrated image"
            )
            
            calibrated_image = st.file_uploader(
                "Upload calibrated image",
                type="png",
                help="Select a PNG file for the calibrated image"
            )

        with col2:
            # Create a container for the comparison view
            st.markdown("<div class='comparison-container'>", unsafe_allow_html=True)
            
            if calibrated_image and uncalibrated_image:
                try:
                    # Open and process images
                    img1 = Image.open(calibrated_image)
                    img2 = Image.open(uncalibrated_image)

                    img1_array = np.array(img1)
                    img2_array = np.array(img2)

                    # Verify image dimensions
                    if img1_array.shape != img2_array.shape:
                        st.error("⚠️ Images must have the same dimensions")
                        return

                    # Extract filenames for labels
                    label1 = calibrated_image.name if calibrated_image else "Calibrated"
                    label2 = uncalibrated_image.name if uncalibrated_image else "Uncalibrated"

                    # Display comparison slider
                    image_comparison(
                        img1=img1_array,
                        img2=img2_array,
                        label1=label1,
                        label2=label2,
                        width=int(img1_array.shape[1] * 0.4),
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True
                    )
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()