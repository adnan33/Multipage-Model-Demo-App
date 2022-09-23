import streamlit as st

style_path = "css/style.css"

main_title = "Model Demo App"

def apply_style():
    # Including bootstrap 
    st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">', unsafe_allow_html=True)

    # Loading the css file.
    with open(style_path) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)