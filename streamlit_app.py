import streamlit as st
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


#   -----------------------------------------------------------------------
logo_path = "./app/static/logo_small.png"
icon = "./app/static/icon.png"


with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

home = st.Page("EDA/home.py", title="Home", icon=":material/home:", default=True)
EDA = st.Page("EDA/EDA.py", title="EDA", icon=":material/table:")  
forecast = st.Page("Forecast/forecast.py", title="Pronóstico", icon=":material/search:")

pg = st.navigation([home,EDA,forecast], expanded=True)

st.set_page_config(page_title='DEMO pronósticos', page_icon=icon)

pg.run()
