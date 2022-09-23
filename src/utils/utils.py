import onnxruntime as rt
from pydantic import BaseSettings
import numpy as np
from PIL import Image
import os
import gdown


class URLConfigs(BaseSettings):
    cloth_seg_model_url = ""
    cloth_demo_image_url = ""
    hp_model_url = ""
    hp_demo_image_url = ""


class EnvKeyConfigs(BaseSettings):
    cloth_model_env_key = "CLOTH_SEG_MODEL_URL"
    cloth_demo_image_env_key = "CLOTH_DEMO_IMAGE_URL"
    hp_model_env_key = "HP_MODEL_URL"
    hp_demo_image_env_key = "HP_DEMO_IMAGE_URL"


class ImageConfigs(BaseSettings):
    hp_model_input_shape = (512, 512)
    hp_output_shape = (768, 1024)
    cloth_image_input_shape = (320, 320)
    cloth_image_output_shape = (384, 540)


class PathConfigs(BaseSettings):
    hp_model_path = "models/human_parsing_model.onnx"
    cloth_seg_model_path = "models/cloth_segmentation_model.onnx"
    cloth_demo_image_path = "demo_images/cloth_test_image.jpg"
    hp_demo_image_path = "demo_images/hp_test_image.jpg"


def download_resource(path, url, env_key):
    if not os.path.isfile(path):
        url = os.environ[env_key]
        gdown.download(url, path, quiet=False)


def preprocess_input(input, input_shape) -> np.array:
    """
    Method for preprocessing the input image to be compatible with the segmentation models.
    Here we resize the input image, convert it from PIL image format to numpy array, reshape the
    input to a tensor appropriate for the models and lastly we convert the image to float32 and normalize it.

    Args:
        input (image): PIL image
        input_shape (tuple): input image shape

    Returns:
        np.array : processed and transformed image ready to be inferenced
    """
    input = input.resize(input_shape)
    input = np.expand_dims(np.array(input).transpose(2, 0, 1), axis=0)
    return input.astype(np.float32) / 255.0


def get_cloth_segmask(input: np.array) -> np.array:
    """
    This method performs inference with the cloth segmentation model and
    returns the predicted cloth segmentation mask.


    Args:
        input (np.array): preprocessed input array ready to be inferenced.

    Returns:
        np.array: predicted segmentation mask.
    """
    input = preprocess_input(input, img_shapes.cloth_image_input_shape)
    ort_input = {clothseg_model.get_inputs()[0].name: input}
    output = clothseg_model.run(None, ort_input)[0][0][0] > 0.75
    return output


def get_hp_segmask(input: np.array) -> np.array:
    """
    This method performs inference with the human parsing model and
    returns the predicted human parsing segmentation mask.

    Args:
        input (np.array): preprocessed input array ready to be inferenced.

    Returns:
        np.array: predicted segmentation mask.
    """
    input = preprocess_input(input, img_shapes.hp_model_input_shape)
    ort_input = {hp_model.get_inputs()[0].name: input}
    output = np.argmax(hp_model.run(None, ort_input)[0][0], 0)
    return output


def get_hp_colormask(input):
    input_shape = input.size
    output = get_hp_segmask(input)
    output = Image.fromarray(np.asarray(output, dtype=np.uint8))
    output.putpalette(colormap.astype(np.uint8).flatten().tolist())
    output = output.resize(input_shape)
    return output


def get_colormask_overlay(input, colormask):
    input = input.convert("RGBA")
    colormask = colormask.convert("RGBA")
    overlay = Image.blend(input, colormask, 0.65)
    return overlay


img_shapes = ImageConfigs()
paths = PathConfigs()
urls = URLConfigs()
env_keys = EnvKeyConfigs()

## Creating directories in case of initial app launch
os.makedirs("models", exist_ok=True)
os.makedirs("demo_images", exist_ok=True)

# downloading the models and demo images for the 1st time
download_resource(
    path=paths.cloth_seg_model_path,
    url=urls.cloth_seg_model_url,
    env_key=env_keys.cloth_model_env_key,
)

download_resource(
    path=paths.hp_model_path, url=urls.hp_model_url, env_key=env_keys.hp_model_env_key
)

download_resource(
    path=paths.cloth_demo_image_path,
    url=urls.cloth_demo_image_url,
    env_key=env_keys.cloth_demo_image_env_key,
)

download_resource(
    path=paths.hp_demo_image_path,
    url=urls.hp_demo_image_url,
    env_key=env_keys.hp_demo_image_env_key,
)


# human parsing colormap
colormap = [
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
colormap = np.ceil(np.array(colormap) * 255.0)

# load models
clothseg_model = rt.InferenceSession(paths.cloth_seg_model_path)
hp_model = rt.InferenceSession(paths.hp_model_path)

