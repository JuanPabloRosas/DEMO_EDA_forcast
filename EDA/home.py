import streamlit as st
import pandas as pd


logo_path = "C:\\Users\\Celula1\\.streamlit\\img\\logo_small.png"
icon = "C:\\Users\\Celula1\\.streamlit\\img\\icon.png"



with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)



st.header('Transformando datos en decisiones inteligentes')
st.subheader('Desarrollamos plataformas de inteligencia ofreciendo analítica avanzada como servicio')
st.divider()
st.subheader('¿Qué hacemos en Datlas')
st.write('Apalancados en técnicas de big data, inteligencia artificial y analítica avanzada convertimos datos en accionables de una manera ágil y sencilla.')
st.markdown('- Contamos con mapas para inteligencia comercial, reportes de mercado automatizados y dashboards de monitoreo')
st.markdown('- Generamos insights integrando la información de tu organización a nuestros modelos')
st.markdown('- Construimos algoritmos y procesos personalizados para ayudarte a revelar oportunidades monetizables')
st.markdown('- Todas las plataformas en Datlas cuentan con control de accesos seguro y personalizado')