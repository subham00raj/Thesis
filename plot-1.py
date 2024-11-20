import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tifffile

def normalize(image, vmin, vmax):
    image = np.abs(image)
    norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return np.clip(norm, vmin, vmax)

def main():
    st.title("TIFF Image Viewer")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Move file uploader to sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a TIFF file", type=['tiff', 'tif'])
    
    # Add separator for visual clarity
    st.sidebar.markdown("---")
    
    # Display settings in sidebar
    st.sidebar.header("Display Settings")
    vmin = st.sidebar.number_input("Min Value", 0.0, 1.0, 0.0001, format="%.4f")
    vmax = st.sidebar.number_input("Max Value", 0.0, 1.0, 0.001, format="%.4f")
    
    if uploaded_file is not None:
        # Read and normalize the TIFF image
        tiff_image = tifffile.imread(uploaded_file)
        normalized_image = normalize(tiff_image, vmin, vmax)
        
        # Create figure
        fig = go.Figure()
        
        # Add the image as a heatmap
        fig.add_trace(
            go.Heatmap(
                z=normalized_image,
                colorscale='Gray',
                showscale=True,
                hoverongaps=False,
                hovertemplate=
                'x: %{x}<br>'+
                'y: %{y}<br>'+
                'value: %{z:.6f}<extra></extra>'
            )
        )
        
        # Update layout with axes turned off
        fig.update_layout(
            width=800,
            height=800,
            xaxis=dict(
                visible=False,
                scaleanchor='y',
                constrain='domain',
                showgrid=False,
            ),
            yaxis=dict(
                visible=False,
                constrain='domain',
                showgrid=False,
                autorange='reversed'
            ),
            margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
            dragmode='pan',
        )
        
        # Configure mouse interactions
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['resetScale2d'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'doubleClick': 'reset',
        }
        
        # Display the plot
        st.plotly_chart(fig, config=config, use_container_width=True)
        
        # Add image information to sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Image Information")
        st.sidebar.write(f"Dimensions: {tiff_image.shape}")
        st.sidebar.write(f"Data Type: {tiff_image.dtype}")
        st.sidebar.write(f"Value Range: [{np.min(tiff_image):.6f} to {np.max(tiff_image):.6f}]")
        
        # Add usage instructions to sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Instructions")
        st.sidebar.info("""
        - Scroll to zoom in/out
        - Drag to pan
        - Double click to reset view
        - Hover for pixel values
        """)
    else:
        # Display upload instructions in main area
        st.info("Please upload a TIFF file using the sidebar to begin")

if __name__ == "__main__":
    main()