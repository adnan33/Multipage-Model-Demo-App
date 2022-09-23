import streamlit as st
from static import *
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
    """## Models/Projects
Following model/project demos are added here. It will be updated after each project. Check out the [github](https://github.com/adnan33/Multipage-Model-Demo-App.git) repository to know more about the projects.

1. **[Cloth Segmentation Model](https://adnan33-multipage-model-demo-app-srcindex-0bvfil.streamlitapp.com/Cloth_Segmentation/?target=_self)**
2. **[Human Parsing Model](https://adnan33-multipage-model-demo-app-srcindex-0bvfil.streamlitapp.com/Human_Parsing/?target=_self)**
3. **[Plant Disease Detection](https://adnan33-plant-disease-detector-app-srcapp-eud76f.streamlitapp.com/)**""",
    unsafe_allow_html=True,
)
