import streamlit as st
from static import *
from utils.utils import paths

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
apply_style()
st.title(main_title)
st.sidebar.success("Select a demo above.")
st.write(
    "This webapp contains the demo inference of the deep learning models that I have trained and worked on for my pet projects."
)
st.markdown(
    """## Models/Projects
Following model/project demos are added here. It will be updated after each project. Check out the [github](https://github.com/adnan33/Multipage-Model-Demo-App.git) repository to know more about the projects.

1. **Cloth Segmentation Model**
2. **Human Parsing Model** """,
    unsafe_allow_html=True,
)
