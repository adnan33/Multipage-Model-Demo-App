import streamlit as st
from statics import apply_style, main_title
from utils.utils import paths

st.set_page_config(
    page_title="Adnan's DL Model Showcase APP",
    page_icon="👋",
)
apply_style()
st.title(main_title)
st.sidebar.success("Select a demo above.")
st.write(
    "This webapp contains the demo inference of the deep learning models that I have trained and worked on for my personal projects."
)
st.markdown(
    f"""## Models/Projects
Following model/project demos are added here. It will be updated after each project. Check out the [github](https://github.com/adnan33/Multipage-Model-Demo-App.git) repository to know more about the projects.

1. <b><a href="Cloth_Segmentation/" target="_self">Cloth Segmentation Model</a></b>
    - A ***binary semantic segmentation*** model to segment out cloth from single cloth image.
2. <b><a href="Cloth_Segmentation_On_Person/" target="_self">Cloth Segmentation On Human Body Model</a></b>
    - A ***multi-class (3 classes) semantic segmentation*** model to segment out upper and lower body cloths from an image of a person.
3. <b><a href="Human_Parsing/" target="_self">Human Parsing Model</a></b>
    -  A ***multi-class (20 classes) semantic segmentation*** model for the task of **[human parsing](https://paperswithcode.com/task/human-parsing)**.
4. **[Plant Disease Detection](https://adnan-plant-disease-detector.streamlitapp.com/)**
    - A mutli class classification model to detect plant diseases from leaf image.
    """,
    unsafe_allow_html=True,
)
