import streamlit as st
import pandas as pd

logo_path = "./app/static/logo_small.png"
icon = "./app/static/icon.png"

with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

# Set the background image
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header('Transformando datos en decisiones inteligentes')
st.video("https://youtu.be/_xOv22BBGi8?si=OiH8IlG8B0EqiB-H")

im1 = "https://datlas-static.s3.amazonaws.com/static/img/sol_Mapas.jpg"
im2 = "https://datlas-static.s3.amazonaws.com/static/img/sol_Laura.jpg"
im3 = "https://datlas-static.s3.amazonaws.com/static/img/sol_Dashboard.jpg"
im4 = "https://datlas-static.s3.amazonaws.com/static/img/sol_Consultoria.jpg"


st.subheader('¿Qué hacemos en Datlas')
st.divider()
st.write('Apalancados en técnicas de big data, inteligencia artificial y analítica avanzada convertimos datos en accionables de una manera ágil y sencilla.')
#st.markdown('- Contamos con mapas para inteligencia comercial, reportes de mercado automatizados y dashboards de monitoreo')
#st.markdown('- Generamos insights integrando la información de tu organización a nuestros modelos')
#st.markdown('- Construimos algoritmos y procesos personalizados para ayudarte a revelar oportunidades monetizables')
#st.markdown('- Todas las plataformas en Datlas cuentan con control de accesos seguro y personalizado')
captions = ['Contamos con mapas para inteligencia comercial, reportes de mercado automatizados y dashboards de monitoreo',
            'Generamos insights integrando la información de tu organización a nuestros modelos',
            'Construimos algoritmos y procesos personalizados para ayudarte a revelar oportunidades monetizables',
            'Todas las plataformas en Datlas cuentan con control de accesos seguro y personalizado']
st.image([im1, im2, im3,im4], width=160, caption= captions)
st.subheader('Blogs de interes:')
st.divider()
st.markdown("- [Inteligencia Artificial en la Predicción de Ventas: Desmitificando la «Caja Negra» - Columna de Opinión Datlas](%s)" % "https://blogdatlas.wordpress.com/2024/10/23/inteligencia-artificial-en-la-prediccion-de-ventas-desmitificando-la-caja-negra-columna-de-opinion-datlas/",)
st.markdown("- [Generación de Pronósticos con Modelos de Series de Tiempo y Cómputo en la Nube | GCP y Nixtla - Columnas de Investigación DATLAS](%s)" % "https://blogdatlas.wordpress.com/2024/06/22/generacion-de-pronosticos-con-modelos-de-series-de-tiempo-y-computo-en-la-nube-gcp-y-nixtla-columnas-de-investigacion-datlas/")
st.markdown("- [Predicciones de series temporales con PROPHET de facebook (META) - Columna de Investigación DATLAS](%s)" % "https://blogdatlas.wordpress.com/2024/05/11/predicciones-de-series-temporales-con-prophet-de-facebook-meta-columna-de-investigacion-datlas/")
st.markdown("- [Utilizando la Inteligencia Artificial para Pronósticos de Inundaciones Globales - Columna de Investigación DATLAS](%s)" % "https://blogdatlas.wordpress.com/2024/03/23/utilizando-la-inteligencia-artificial-para-pronosticos-de-inundaciones-globales-columna-de-investigacion-datlas/")
