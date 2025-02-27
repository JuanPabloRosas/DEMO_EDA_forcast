import streamlit as st
import streamlit.components.v1 as components

#   DOCKER
logo_path = "static/logo_small.png"
icon = "static/icon.png"
img_header = "static/eda_header.png"
video = 'https://player.vimeo.com/video/1058312363'

with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.image(img_header)
components.iframe(video, height=500)

im1 = "https://blogdatlas.wordpress.com/wp-content/uploads/2024/10/image-3.png?w=900"
im2 = "https://blogdatlas.wordpress.com/wp-content/uploads/2024/06/banner-1.png?w=900"
im3 = "https://datlas-static.s3.amazonaws.com/static/img/sol_Dashboard.jpg"
im4 = "https://datlas-static.s3.amazonaws.com/static/img/sol_Consultoria.jpg"


st.subheader('Blogs de interes:')
st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
st.markdown("- [Inteligencia Artificial en la Predicción de Ventas: Desmitificando la «Caja Negra»](%s)" % "https://blogdatlas.wordpress.com/2024/10/23/inteligencia-artificial-en-la-prediccion-de-ventas-desmitificando-la-caja-negra-columna-de-opinion-datlas/")
st.markdown("- [Generación de Pronósticos con Modelos de Series de Tiempo y Cómputo en la Nube | GCP y Nixtla](%s)" % "https://blogdatlas.wordpress.com/2024/06/22/generacion-de-pronosticos-con-modelos-de-series-de-tiempo-y-computo-en-la-nube-gcp-y-nixtla-columnas-de-investigacion-datlas/")
st.markdown("- [Predicciones de series temporales con PROPHET de facebook (META)](%s)" % "https://blogdatlas.wordpress.com/2024/05/11/predicciones-de-series-temporales-con-prophet-de-facebook-meta-columna-de-investigacion-datlas/")
st.markdown("- [Utilizando la Inteligencia Artificial para Pronósticos de Inundaciones Globales](%s)" % "https://blogdatlas.wordpress.com/2024/03/23/utilizando-la-inteligencia-artificial-para-pronosticos-de-inundaciones-globales-columna-de-investigacion-datlas/")