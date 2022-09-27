import streamlit as st
from pydantic import BaseSettings

style_path = "css/style.css"

main_title = "Model Demo App"

# human parsing colormap
hp_colormap_list = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.99609375, 0.0, 0.0],
    [0.0, 0.33203125, 0.0],
    [0.6640625, 0.0, 0.19921875],
    [0.99609375, 0.33203125, 0.0],
    [0.0, 0.0, 0.33203125],
    [0.0, 0.46484375, 0.86328125],
    [0.33203125, 0.33203125, 0.0],
    [0.0, 0.33203125, 0.33203125],
    [0.33203125, 0.19921875, 0.0],
    [0.203125, 0.3359375, 0.5],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.99609375],
    [0.19921875, 0.6640625, 0.86328125],
    [0.0, 0.99609375, 0.99609375],
    [0.33203125, 0.99609375, 0.6640625],
    [0.6640625, 0.99609375, 0.33203125],
    [0.99609375, 0.99609375, 0.0],
    [0.99609375, 0.6640625, 0.0],
]
# On person cloth segmentation colormap
opcs_colormap_list = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.99609375, 0.99609375],
]


class URLConfigs(BaseSettings):
    cloth_seg_model_url = ""
    cloth_demo_image_url = ""
    opcloth_seg_model_url = ""
    hp_model_url = ""
    hp_demo_image_url = ""


class EnvKeyConfigs(BaseSettings):
    cloth_model_env_key = "CLOTH_SEG_MODEL_URL"
    opcloth_model_env_key = "OPCLOTH_SEG_MODEL_URL"
    cloth_demo_image_env_key = "CLOTH_DEMO_IMAGE_URL"
    hp_model_env_key = "HP_MODEL_URL"
    hp_demo_image_env_key = "HP_DEMO_IMAGE_URL"


class ImageConfigs(BaseSettings):
    hp_model_input_shape = (512, 512)
    hp_output_shape = (768, 1024)
    cloth_model_input_shape = (320, 320)
    cloth_image_output_shape = (384, 540)


class PathConfigs(BaseSettings):
    hp_model_path = "models/human_parsing_model.onnx"
    cloth_seg_model_path = "models/cloth_segmentation_model.onnx"
    opcloth_seg_model_path = "models/op_cloth_segmentation_model.onnx"
    cloth_demo_image_path = "demo_images/cloth_test_image.jpg"
    hp_demo_image_path = "demo_images/hp_test_image.jpg"


def apply_style():
    # Including bootstrap
    st.markdown(
        '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )

    # Loading the css file.
    with open(style_path) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
