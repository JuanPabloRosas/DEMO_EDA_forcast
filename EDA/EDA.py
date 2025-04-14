import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
pd.options.display.max_columns = None
import numpy as np
from scipy import stats
import gc

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

#   DOCKER
logo_path = "static/logo_small.png"
icon = "static/icon.png"
img_eda = "static/EDA & Forecast_bl.png"

with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#   -----------------------------------------------------------------------
def read_file(filename):
    try:
        db = pd.read_csv(filename, encoding='utf-8')
    except:
        try:
            db = pd.read_csv(filename, encoding='utf-8-sig')
        except:
            db = pd.read_csv(filename, encoding='latin1')
    
    db.columns = db.columns.str.upper()
    return db

def corr(db):
    plt.figure(figsize=(8, 4))
    sns.heatmap(db.select_dtypes(exclude='object').corr(), cmap='viridis', annot = True)
    plt.title('Heatmap Correlación')
    plt.tight_layout()
    plt.show()

def c_nan(db):
    nan = pd.DataFrame(db.isna().sum())
    nan.columns = ['Nan']
    return nan

def c_cero(db):
    cero = pd.DataFrame(pd.DataFrame(db == 0).sum())
    cero.columns = ['Cero']
    #cero = cero.to_html()
    return cero

def c_negative(db):
    neg = pd.DataFrame(pd.DataFrame(db.select_dtypes(exclude=['datetime64','object']) < 0).sum())
    neg.columns = ['Negativos']
    return neg

def c_duplicate(db):
    d = {}
    for columna in db.columns:
        dup = db[db[columna].duplicated()]
        if not dup.empty:
            d[columna] = len(dup)

    return pd.DataFrame(d.items(), columns=['Columna', 'Duplicados'])

def identify_outliers(db, m):
    if(m == 'zscore'):
        z = np.abs(stats.zscore(db['y']))
        # Identify outliers as students with a z-score greater than 3 standar desviation
        threshold = 3
        outliers = db[z > threshold]
    if(m == 'iqr'): #   interquartile range
        # calculate IQR for column Height
        Q1 = db[['y']].quantile(0.25)
        Q3 = db[['y']].quantile(0.75)
        IQR = Q3 - Q1
        # identify outliers
        threshold = 3
        outliers = db[((db['y'] < Q1[0] - threshold * IQR[0]) | (db['y'] > Q3[0]- threshold * IQR[0]))]
    return outliers

def rfm(df):
    df.columns = ['fecha','cliente','cantidad','precio','producto']
    df_grouped = df.groupby(['fecha','producto','cliente'], as_index=False).sum()
    clientes = list(df_grouped['cliente'].unique())
    df_rfm = pd.DataFrame()
    list_ac = list()
    for cliente in clientes:
        #   cliente = 'Technics Stores Inc.'
        #   Compra mas reciente
        r = pd.to_datetime(df_grouped[df_grouped['cliente'] == cliente]['fecha']).max()

        #   Cuantas veces ha comprado
        f = len(pd.to_datetime(df_grouped[df_grouped['cliente'] == cliente]['fecha']))
        
        #   Cuanto ha gastado
        temp = df_grouped[df_grouped['cliente'] == cliente]
        temp['total'] = temp['cantidad'] * temp['precio']
        m = temp['total'].mean()
        list_ac.append([cliente,r,f,m])
    df_rfm = pd.DataFrame(list_ac)
    df_rfm.columns = ['cliente','R','F','M']
    return df_rfm
#   -----------------------------------------------------------------------

st.subheader('¿Qué es un EDA?')
st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
st.markdown("""
            Un :blue-background[**EDA (Exploratory Data Analysis, por sus siglas en inglés)**] es un enfoque para analizar conjuntos de datos con el fin de resumir sus principales características, 
            a menudo con el uso de métodos gráficos y estadísticos. El objetivo del EDA es comprender la estructura, las relaciones y las posibles anomalías en los datos antes de aplicar modelos predictivos o realizar inferencias más complejas.
            Algunas de las técnicas comunes en el EDA incluyen:
            
            > - **Estadísticas descriptivas**: Cálculo de medidas como la media, mediana, desviación estándar, percentiles, etc.
            > - **Visualización de datos**: Usar gráficos como histogramas, diagramas de dispersión, diagramas de barras, etc., para identificar patrones, tendencias y posibles outliers.
            > - **Detección de valores atípicos (outliers)**: Identificar puntos de datos que se alejan de manera significativa del resto del conjunto.
            > - **Análisis de correlaciones**: Verificar las relaciones entre variables utilizando matrices de correlación o gráficos de dispersión.
            
            El EDA es crucial para comprender los datos y tomar decisiones informadas antes de aplicar modelos estadísticos o algoritmos de aprendizaje automático.
            """)

st.image(img_eda)

st.write("Para usar el EDA puedes usar alguno de los dataset que tenemos para pruebas o cargar un archivo de datos tabular separado por comas y con encabezados, que sea :blue-background[**menor o igual a 200 Mb**].")

st.header('Análisis exploratorio de datos')
dataset = st.radio(
    "Selecciona un dataset de prueba",
    ["Datos de aerolinea", "Datos de ventas", "Cargar un archivo"],
    captions=[
        "Datos de fechas de vuelos y cantidad de pasajeros",
        "Datos de transacciones de diferentes productos",
        "Archivo csv menor a 200 mb",
    ], index=None
)
if(dataset  == 'Datos de aerolinea'):  #   AEROLINEA
    st.session_state['respaldo'] = read_file('Forecast/airline_passengers.csv')
    st.session_state['respaldo']['MONTH'] = pd.to_datetime(st.session_state['respaldo']['MONTH'])
    st.session_state['datos'] = st.session_state['respaldo']
elif(dataset == 'Datos de ventas'):  #   VENTAS
    st.session_state['respaldo'] = read_file('Forecast/sales_data_sample2.csv')
    st.session_state['datos'] = st.session_state['respaldo']
elif(dataset =="Cargar un archivo"):
    _file = st.file_uploader("Carga un archivo separado por comas (.csv)", type="csv")
    if _file is not None:
        st.session_state['respaldo'] = read_file(_file)
        st.session_state['datos'] = st.session_state['respaldo']
#   -----------------------------------------------------------------------
#   ESTADISTICA BASICA
if 'datos' in st.session_state:    
#if _file is not None:    
    st.subheader('Visualización de Datos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)    
    st.write(st.session_state['datos'])
    
    st.subheader('Renglones y columnas')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.write('El archivo tiene :blue-background[**' + str(st.session_state['datos'].shape[0]) + ' renglones y ' + str(st.session_state['datos'].shape[1]) +
              ' columnas**]. Además, en la siguiente tabla podemos ver cuantos datos unicos hay por cada columna.')
    
    st.text(st.session_state['datos'].nunique())
    
    
    st.subheader('Duplicados, ceros, negativos, Nan y tipos de dato')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)    
    st.write('En la siguiente tabla se muestra la cantidad de :blue-background[**datos duplicados o negativos por columna**], así como la cantidad de :blue-background[**ceros o datos faltantes (Nan) y el tipo de dato**]')
    nan_ = c_nan(st.session_state['datos'])
    cero_=  c_cero(st.session_state['datos'])
    neg_=  c_negative(st.session_state['datos'])
    dup_=  c_duplicate(st.session_state['datos'])
    info = pd.DataFrame(st.session_state['datos'].dtypes)
    info.columns = ['Tipo de dato']
    ac = pd.concat([cero_,neg_,nan_,info], axis=1)
    ac.style.format_index(str.upper, axis = 1)
    st.write(dup_.merge(ac,right_on=ac.index, left_on='Columna', how='right'))
    
    st.subheader('Estadística básica')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)    
    st.write(st.session_state['datos'].describe())
#   ----------------------------------------------------------------------------------------------
    st.subheader('Gráfica de barras')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)    
    st.write('Selecciona los valores para el eje horizontal y para el eje vertical. La característica que elijas para el eje vertical se va a agrupar y sumar, y se van a mostrar mediante un grafico de barras.')
    col1, col2 = st.columns([1,1])
    with col1:
        x_bar = st.selectbox('Eje horizontal bar plot: ',st.session_state['datos'].columns, index=None, placeholder='Selecciona una opción')
        y_bar = st.selectbox('Eje vertical bar plot: ',st.session_state['datos'].columns, index=None, placeholder='Selecciona una opción')
    with col2:
        operation_bar = st.radio('Operación de agrupamiento', ['Sumar', 'Contar'], key='radio_bar')
    if((x_bar is not None) & (y_bar is not None)):
        if(operation_bar == 'Sumar'):
            df_temp = st.session_state['datos'].groupby([x_bar], as_index=False).sum()
        else:
            df_temp = st.session_state['datos'].groupby([x_bar], as_index=False).count()
        
        try:
            fig = px.bar(df_temp, x=x_bar, y=y_bar)
            st.plotly_chart(fig,use_container_width=True)
        except:
            st.error('Elige una columna que se pueda sumar para el eje vertical o elige contar como operación de agrupamiento...')
#   ----------------------------------------------------------------------------------------------
    st.subheader('Gráfica de lineas')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)    
    st.write('Selecciona los valores para el eje horizontal y para el eje vertical. La característica que elijas para el eje vertical se va a agrupar y sumar, y se van a mostrar mediante un grafico de lineas.')
    col1, col2 = st.columns([1,1])
    with col1:
        x_line = st.selectbox('Eje horizontal line plot: ',st.session_state['datos'].columns, index=None, placeholder='Selecciona una opción')
        y_line = st.selectbox('Eje vertical line plot: ',st.session_state['datos'].columns, index=None, placeholder='Selecciona una opción')
    with col2:
        operation_line = st.radio('Operación de agrupamiento', ['Sumar', 'Contar'], key='radio_line')
    if((x_line is not None) & (y_line is not None)):
        df_temp = st.session_state['datos'].groupby([x_line], as_index=False).sum()
        fig = px.line(df_temp, x=x_line, y=y_line, title="Line plot")
        st.plotly_chart(fig,use_container_width=True)
#   ----------------------------------------------------------------------------------------------
    st.subheader('Gráfica de Correlación')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.write('El grafico de correlación te puede ayudar a visualizar si existe una relación positiva (Un número entre cero y uno) o una relación negativa (Un número entre menos uno y cero).' +
             ' Si la relación es positiva indica que si una de las características aumenta su valor, la otra también. Si la relación e snegativa indica lo contrario.')
    fig = px.imshow(round(st.session_state['datos'].select_dtypes(exclude='object').corr(),3), text_auto=True, aspect='auto')
    st.plotly_chart(fig,use_container_width=True)
#   ----------------------------------------------------------------------------------------------
    st.subheader('Histograma')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    y_hist = st.selectbox('Histograma: ',st.session_state['datos'].columns, index= 0)
    hist_fig = px.histogram(st.session_state['datos'], x=y_hist)
    st.plotly_chart(hist_fig,use_container_width=True)
#   ----------------------------------------------------------------------------------------------
    st.subheader('Gráfia de disperción')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        x = st.selectbox('Eje horizontal scatter plot: ',st.session_state['datos'].columns, index=None, placeholder='Selecciona una opción')
        y = st.selectbox('Eje vertical scatter plot: ',st.session_state['datos'].columns, index=None, placeholder='Selecciona una opción')
    with col2:
        operation_scatter = st.radio('Operación de agrupamiento', ['Sumar', 'Contar'], key='radio_scatter')
    if((x is not None) & (y is not None)):
        df_temp = st.session_state['datos'].groupby([x], as_index=False).sum()
        fig = px.scatter(df_temp, x=x, y=y, title="Scatter plot")
        st.plotly_chart(fig,use_container_width=True)
    #   ------------------------------- ANALISIS RFM ---------------------------------------------------------------
    
    #st.subheader('Análisis RFM')
    #st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    #col1, col2 = st.columns([1,1])
    #with col1:
    #    fechas_compra = st.selectbox('Fechas de compra: ',st.session_state['datos'].columns, index=None)
    #    clientes = st.selectbox('Clientes: ',st.session_state['datos'].columns, index=None)
    #with col2:
    #    cantidad = st.selectbox('Cantidad de compra: ',st.session_state['datos'].columns, index=None)
    #    costo = st.selectbox('Costos de compra: ',st.session_state['datos'].columns, index=None)
    #    producto = st.selectbox('Productos: ',st.session_state['datos'].columns, index=None)
    
    #if((fechas_compra is not None) & (clientes is not None) & (cantidad is not None) & (costo is not None) & (producto is not None)):
    #    _rfm = rfm(st.session_state['datos'][[fechas_compra,clientes,cantidad,costo,producto]])
    #    st.write(_rfm)
    #   ----------------------------------------------------------------------------------------------

    del nan_
    del cero_
    del neg_
    del dup_
    del ac
    del x
    gc.collect()

gc.collect()
