import streamlit as st

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


#   -----------------------------------------------------------------------
logo_path = "C:\\Users\\Celula1\\.streamlit\\img\\logo_small.png"
icon = "C:\\Users\\Celula1\\.streamlit\\img\\icon.png"

with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

home = st.Page("EDA/home.py", title="Home", icon=":material/home:", default=True)
EDA = st.Page("EDA/EDA.py", title="EDA", icon=":material/table:")
forecast = st.Page("Forecast/forecast.py", title="Pronóstico", icon=":material/search:")

#pg = st.navigation({"Inicio": [home,EDA],"Tools": [forecast]})
pg = st.navigation([home,EDA,forecast])

st.set_page_config(page_title='DEMO pronósticos', page_icon=icon)

pg.run()

