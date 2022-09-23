from io import BytesIO
import streamlit as st
from utils.utils import *


def cloth_segmentation_process(img, output_shape):
    st.markdown("<h3 style=' color: #666;'>Input Image</h3>", unsafe_allow_html=True)
    if output_shape:
        st.image(
            img.resize(img_shapes.cloth_image_output_shape),
            caption="Input Cloth Image",
            channels="RGB",
        )
    else:
        st.image(img, caption="Input Cloth Image", channels="RGB")
    output = get_cloth_segmask(img)
    output = Image.fromarray((output * 255).astype(np.uint8))
    st.markdown(
        "<h3 style=' color: #666;'>Predicted Segmentation Mask</h3>",
        unsafe_allow_html=True,
    )
    if output_shape:
        st.image(
            output.resize(img_shapes.cloth_image_output_shape),
            caption="Predicted Cloth Segmentation Mask",
            channels="RGB",
        )
    else:
        output = output.resize(img.size)
        st.image(output, caption="Predicted Cloth Segmentation Mask", channels="RGB")


def human_parsing_segmentation_process(img, output_shape):
    st.markdown("<h3 style=' color: #666;'>Input Image</h3>", unsafe_allow_html=True)
    if output_shape:
        st.image(img, caption="Input image", channels="RGB")
    else:
        st.image(
            img.resize(img_shapes.hp_output_shape),
            caption="Input image",
            channels="RGB",
        )
    output_mask = get_hp_colormask(img)
    st.markdown(
        "<h3 style=' color: #666;'>Predicted Segmentation Mask</h3>",
        unsafe_allow_html=True,
    )
    if output_shape:
        st.image(output_mask, caption="Predicted Segmentation Mask", channels="RGB")
    else:
        st.image(
            output_mask.resize(img_shapes.hp_output_shape),
            caption="Predicted Segmentation Mask",
            channels="RGB",
        )
    output = get_colormask_overlay(img, output_mask)
    st.markdown(
        "<h3 style=' color: #666;'>Predicted Segmentation Mask Overlay</h3>",
        unsafe_allow_html=True,
    )
    if output_shape:
        st.image(
            output, caption="Predicted Segmentation Mask Overlayed", channels="RGB"
        )
    else:
        st.image(
            output.resize(img_shapes.hp_output_shape),
            caption="Predicted Segmentation Mask Overlayed",
            channels="RGB",
        )


def demo_cloth_segmentation(output_shape):
    img = Image.open(paths.cloth_demo_image_path)
    cloth_segmentation_process(img, output_shape)


def custom_cloth_segmentation(uploaded_file, output_shape):
    img = uploaded_file.read()
    img = Image.open(BytesIO(img))
    cloth_segmentation_process(img, output_shape)


def demo_human_parsing_segmentation(output_shape):
    img = Image.open(paths.hp_demo_image_path)
    human_parsing_segmentation_process(img, output_shape)


def custom_human_parsing_segmentation(uploaded_file, output_shape):
    img = uploaded_file.read()
    img = Image.open(BytesIO(img))
    human_parsing_segmentation_process(img, output_shape)
