# Multipage-Model-Demo-App
A multipage demo inference application of different deep learning models built with Streamlit library. The app is built using **Python 3.9.13**.
Here's the [app](https://adnan33-multipage-model-demo-app-srcindex-0bvfil.streamlitapp.com/)

## Models/Projects
Following model/project demos are added here. It will be updated after each project.

1. **Cloth Segmentation Model**
2. **Cloth Segmentation On Human Body Model**
3. **Human Parsing Model** 

## Model/Project details
Here are some brief details of each model/project:
### Cloth Segmentation Model
##### Dataset
This is a *binary semantic segmentation* model trained on a dataset of single clothing article images. To specify, I used the 
cloth images and masks from the [VITON PLUS](https://www.kaggle.com/datasets/rkuo2000/viton-plus) dataset.
##### Model
 I used the **MANet-EfficientnetB2** architecture for this and trained it for 40 epochs on the dataset. The model metrics are:
```
  'test_DICE': 0.992,
  'test_IOU': 0.984,
  'test_TREVSKY': 0.989,
  'test_loss': 0.0265
```
The model is trained with **pytorch** and converted to **onnx** (with optimization) to speed up inference.

### Cloth Segmentation On Human Body Model
##### Dataset
This is a *multi class semantic segmentation* model to segment cloths from upper and lower part of the body . For this model, I have used the [DeepFashion2](https://paperswithcode.com/dataset/deepfashion2) dataset. I have modified this dataset to have only upper and lower body cloths rather than the original dataset labels. This version of the dataset has 126750 training,3250 validation and 10000 test person images and corresponding pixel level annotations. The dataset has 3 different labels. The class labels are *background*,*upper_body_cloth* and *lower_body_cloth*.

##### Model
Current model is based on **UNET++-EfficientnetB3** architecture and trained for 18 epochs with the *Categorical cross-entropy* loss function. The model is trained with **pytorch** and converted to **onnx** (with optimization) to speed up inference. the current model metrics are:
```
  'test_IOU': 0.746,
  'test_DICE': 0.841,
  'test_loss': 0.1441
```

### Human Parsing Model
##### Dataset
This is a *multi class semantic segmentation* model to solve the human parsing task. For this model, I have used the
[Look Into Person (LIP)](https://gas.graviti.com/dataset/graviti/LIP) dataset. This version of the dataset has 25000 training,
2500 validation and 2500 test multi person images and corresponding pixel level annotations. The dataset has 20 different human parts annotated. The class 
labels are given below. You can read more about it from [here](https://competitions.codalab.org/competitions/23433).
![image](images/lip_class_details.JPG)
##### Model
Current model is based on **UNET++-EfficientnetB3** architecture and trained for 24 epochs. The model is trained with **pytorch** and converted to **onnx** (with optimization) to speed up inference.\
**Note: This project is still ongoing so the metrics aren't provided as of now.**
