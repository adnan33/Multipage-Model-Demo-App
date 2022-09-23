import streamlit as st
from static import *
from utils.process import demo_cloth_segmentation, custom_cloth_segmentation

title = "Cloth Segmentation Demo"
st.set_page_config(
    page_title=title,
    page_icon="🥻",
)
apply_style()
st.sidebar.success("Select a demo above.")
st.title(title)
options_dict = {"Demo Image": 1, "My Own Image": 0}
method_select = st.selectbox(
    "How would you like to test the model?", ["Demo Image", "My Own Image"], index=0
)
if options_dict[method_select]:
    output_shape = st.checkbox("Output image shape is equal to input image shape")
    demo_cloth_segmentation(output_shape)
else:
    uploaded_file = st.file_uploader(
        "Please Input Cloth Images", type=["jpg", "png", "jpeg", "bmp"]
    )
    output_shape = st.checkbox("Output image shape is equal to input image shape")
    if uploaded_file != None and uploaded_file != []:
        custom_cloth_segmentation(uploaded_file, output_shape)
