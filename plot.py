import streamlit as st
from PIL import Image
import numpy as np
from streamlit_image_comparison import image_comparison
from osgeo import gdal
import tempfile
import os
import contextlib
import matplotlib.pyplot as plt
gdal.UseExceptions()

# Set Streamlit page configuration to wide layout and custom page height
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Label styles */
    .image-comparison-label {
        background-color: rgba(0, 0, 255, 0.8); /* Semi-transparent blue */
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
    
    /* Custom button style */
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

def normalize(image, vmin, vmax):
    image = np.abs(image)
    norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return np.clip(norm, vmin, vmax)

@contextlib.contextmanager
def temporary_file(suffix='.tiff'):
    """Context manager for creating and cleaning up a temporary file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.close()
        yield temp_file.name
    finally:
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass

def process_image(uploaded_file, vmin, vmax, output_file):
    """Process a single image file and save it as a PNG."""
    with temporary_file() as temp_path:
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        dataset = gdal.Open(temp_path)
        if dataset is None:
            raise ValueError(f"Unable to open {uploaded_file.name}")
        
        try:
            array = dataset.GetRasterBand(1).ReadAsArray()
            normalized = normalize(array, vmin, vmax)

            # Save the processed array as a PNG file
            plt.imshow(normalized, cmap='gray')
            plt.axis('off')  # Hide axes
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=500)
        finally:
            dataset = None  # This closes the dataset

    return output_file

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
                type="tiff",
                help="Select a TIFF file for the uncalibrated image"
            )
            
            calibrated_image = st.file_uploader(
                "Upload calibrated image",
                type="tiff",
                help="Select a TIFF file for the calibrated image"
            )

            vmin = st.number_input("Minimum Threshold", value=0.0, min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
            vmax = st.number_input("Maximum Threshold", value=1.0, min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")

            # Submit button
            submit_button = st.button("Process and Display Images")

        with col2:
            # Create a container for the comparison view
            st.markdown("<div class='comparison-container'>", unsafe_allow_html=True)
            
            if submit_button and calibrated_image and uncalibrated_image:
                try:
                    with st.spinner('Processing images...'):
                        # Create temporary PNG files for the processed images
                        calibrated_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                        uncalibrated_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

                        # Process and save images
                        calibrated_image_path = process_image(calibrated_image, vmin, vmax, calibrated_image_path)
                        uncalibrated_image_path = process_image(uncalibrated_image, vmin, vmax, uncalibrated_image_path)

                        # Extract filenames for labels
                        label1 = calibrated_image.name if calibrated_image else "Calibrated"
                        label2 = uncalibrated_image.name if uncalibrated_image else "Uncalibrated"

                        # Load images using PIL for comparison
                        img1 = Image.open(calibrated_image_path)
                        img2 = Image.open(uncalibrated_image_path)

                        # Display comparison slider
                        image_comparison(
                            img1=img1,
                            img2=img2,
                            label1=label1,
                            label2=label2,
                            width=650,
                            show_labels=True,
                            make_responsive=True,
                            in_memory=True
                        )
                        
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")
            elif submit_button:
                st.warning("Please upload both calibrated and uncalibrated images before processing.")
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
