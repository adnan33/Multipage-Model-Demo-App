import onnxruntime as rt
import numpy as np
from PIL import Image
import os
import gdown
from statics import (
    URLConfigs,
    EnvKeyConfigs,
    ImageConfigs,
    PathConfigs,
    hp_colormap_list,
    opcs_colormap_list,
)


def download_resource(path: str, url: str, env_key: str):
    """Method to download the largers resource files from google drive. The drive links are saved
    in the secret file.

    Args:
        path (str): Save path of the file
        url (str): URL of the file
        env_key (str): Environment key to get the preset URL
    """
    if not os.path.isfile(path):
        url = os.environ[env_key]
        gdown.download(url, path, quiet=False)


def get_colormask_overlay(input: Image, colormask: Image):
    """Method to overlay the colormask for multiclass segmentation models on the input image.

    Args:
        input (Image): Input image.
        colormask (Image): Generated Color Mask.

    Returns:
        Image: Generated overlay image.
    """
    input = input.convert("RGBA")
    colormask = colormask.convert("RGBA")
    overlay = Image.blend(input, colormask, 0.65)
    return overlay


def preprocess_input(input, input_shape, imagenet_normalization=True) -> np.array:
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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    input = input.resize(input_shape)

    if imagenet_normalization:
        input = ((np.array(input) / 255.0 - np.array(mean)) / np.array(std)).astype(
            np.float32
        )
        input = np.expand_dims(np.array(input).transpose(2, 0, 1), axis=0)
    else:
        input = np.expand_dims(np.array(input).transpose(2, 0, 1), axis=0) / 255.0

    return input.astype(np.float32)


def get_binary_segmask(input: np.array, imagenet_normalization=False) -> np.array:
    """
    This method performs inference with binary segmentation model and
    returns the predicted binary segmentation mask.

    Args:
        input (np.array): preprocessed input array ready to be inferenced.

    Returns:
        np.array: predicted segmentation mask.
    """
    input = preprocess_input(
        input, img_shapes.cloth_model_input_shape, imagenet_normalization
    )
    ort_input = {clothseg_model.get_inputs()[0].name: input}
    output = clothseg_model.run(None, ort_input)[0][0][0] > 0.75
    return output


def get_multiclass_segmask(
    input: np.array, model, task=None, imagenet_normalization=True
) -> np.array:
    """
    This method performs inference with the human parsing model and
    returns the predicted human parsing segmentation mask.

    Args:
        input (np.array): preprocessed input array ready to be inferenced.

    Returns:
        np.array: predicted segmentation mask.
    """
    if task == "HPS":
        input = preprocess_input(
            input, img_shapes.hp_model_input_shape, imagenet_normalization
        )
    else:
        input = preprocess_input(
            input, img_shapes.cloth_model_input_shape, imagenet_normalization
        )
    ort_input = {model.get_inputs()[0].name: input}
    output = np.argmax(model.run(None, ort_input)[0][0], 0)
    return output


def get_hp_colormask(input: np.array):
    """
    Method to generate segmentation mask for the human parsing task.

    Args:
        input (np.array): Input image to be segmented.

    Returns:
        Image: Generated segmentation mask.
    """
    input_shape = input.size
    output = get_multiclass_segmask(input, hp_model, "HPS")
    output = Image.fromarray(np.asarray(output, dtype=np.uint8))
    output.putpalette(colormap.astype(np.uint8).flatten().tolist())
    output = output.resize(input_shape)
    return output


def get_opcs_colormask(input: np.array):
    """
    Method to generate segmentation mask for the cloth segmentation on human body task.

    Args:
        input (np.array): Input image to be segmented.

    Returns:
        Image: Generated segmentation mask.
    """
    input_shape = input.size
    output = get_multiclass_segmask(input, opclothseg_model, "OPCS", False)
    output = Image.fromarray(np.asarray(output, dtype=np.uint8))
    output.putpalette(opcs_colormap.astype(np.uint8).flatten().tolist())
    output = output.resize(input_shape)
    return output


## Creating the static paths as a object
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
    path=paths.opcloth_seg_model_path,
    url=urls.opcloth_seg_model_url,
    env_key=env_keys.opcloth_model_env_key,
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
colormap = np.ceil(np.array(hp_colormap_list) * 255.0)
opcs_colormap = np.ceil(np.array(opcs_colormap_list) * 255.0)

# load models
clothseg_model = rt.InferenceSession(paths.cloth_seg_model_path)
opclothseg_model = rt.InferenceSession(paths.opcloth_seg_model_path)
hp_model = rt.InferenceSession(paths.hp_model_path)
