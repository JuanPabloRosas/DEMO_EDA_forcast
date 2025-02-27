import streamlit as st
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


#   -----------------------------------------------------------------------
#   DOCKER
icon = "static/icon.png"

home = st.Page("EDA/home.py", title="Inicio", icon=":material/home:", default=True)
EDA = st.Page("EDA/EDA.py", title="EDA", icon=":material/table:")  
forecast = st.Page("Forecast/forecast.py", title="Pronóstico", icon=":material/search:")
pg = st.navigation([home,EDA,forecast], expanded=True)

st.set_page_config(page_title='EDA & Forecast', page_icon=icon)

pg.run()
