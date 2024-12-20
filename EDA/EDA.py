import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
os.environ['NIXTLA_ID_AS_COL'] = '1'
pd.options.display.max_columns = None
import numpy as np
from scipy import stats
import gc

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


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

#   -----------------------------------------------------------------------
logo_path = "C:\\Users\\Celula1\\.streamlit\\img\\logo_small.png"
icon = "C:\\Users\\Celula1\\.streamlit\\img\\icon.png"

with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

st.title('Análisis exploratorio de datos')
with st.expander('Selecciona los datos a analizar', expanded=False):
    _file = st.file_uploader("Carga un archivo separado por comas (.csv)", type="csv")
    if _file is not None:
        df = read_file(_file)
        st.session_state['respaldo'] = df
        if('datos' not in st.session_state):
            st.session_state['datos'] = df
    else:
        df = pd.DataFrame()
del df
gc.collect()
#   -----------------------------------------------------------------------
#   ESTADISTICA BASICA
if _file is not None:    
    st.divider()
    st.subheader('Visualización de Datos')
    st.write(st.session_state['datos'])
    st.divider()
    st.subheader('Renglones y columnas')
    st.write('El archivo tiene ' + str(st.session_state['datos'].shape[0]) + ' renglones y ' + str(st.session_state['datos'].shape[1]) +
              ' columnas. Además, en la siguiente tabla podemos ver cuantos datos unicos hay por cada columna.')
    
    st.text(st.session_state['datos'].nunique())
    
    st.divider()
    st.subheader('Duplicados, ceros, negativos, Nan y tipos de dato')
    st.write('En la siguiente tabla se muestra la cantidad de datos duplicados o negativos por columna, así como la cantidad de ceros o datos faltantes (Nan) y el tipo de dato')
    nan_ = c_nan(st.session_state['datos'])
    cero_=  c_cero(st.session_state['datos'])
    neg_=  c_negative(st.session_state['datos'])
    dup_=  c_duplicate(st.session_state['datos'])
    info = pd.DataFrame(st.session_state['datos'].dtypes)
    info.columns = ['Tipo de dato']
    ac = pd.concat([cero_,neg_,nan_,info], axis=1)
    ac.style.format_index(str.upper, axis = 1)
    st.write(dup_.merge(ac,right_on=ac.index, left_on='Columna', how='right'))
    
    st.divider()
    st.subheader('Estadística básica')
    st.write(st.session_state['datos'].describe())

    st.divider()
    st.subheader('Grafico de barras')
    st.write('Selecciona los valores para el eje horizontal y para el eje vertical. La característica que elijas para el eje vertical se va a agrupar y sumar, y se van a mostrar mediante un grafico de barras.')
    col1, col2 = st.columns([1,1])
    with col1:
        x_bar = st.selectbox('Eje horizontal bar plot: ',st.session_state['datos'].columns, index=None)
    with col2:
        y_bar = st.selectbox('Eje vertical bar plot: ',st.session_state['datos'].columns, index=None)
    if((x_bar is not None) & (y_bar is not None)):
        df_temp = st.session_state['datos'].groupby([x_bar], as_index=False).sum()
        fig = px.bar(df_temp, x=x_bar, y=y_bar)
        st.plotly_chart(fig,use_container_width=True)
    
    st.divider()
    st.subheader('Line plot')
    st.write('Selecciona los valores para el eje horizontal y para el eje vertical. La característica que elijas para el eje vertical se va a agrupar y sumar, y se van a mostrar mediante un grafico de lineas.')
    col1, col2 = st.columns([1,1])
    with col1:
        x_line = st.selectbox('Eje horizontal line plot: ',st.session_state['datos'].columns, index=None)
    with col2:
        y_line = st.selectbox('Eje vertical line plot: ',st.session_state['datos'].columns, index=None)
    if((x_line is not None) & (y_line is not None)):
        df_temp = st.session_state['datos'].groupby([x_line], as_index=False).sum()
        fig = px.line(df_temp, x=x_line, y=y_line, title="Test line plot")
        st.plotly_chart(fig,use_container_width=True)

    st.divider()
    st.subheader('Correlación')
    st.write('El grafico de correlación te puede ayudar a visualizar si existe una relación positiva (Un número entre cero y uno) o una relación negativa (Un número entre menos uno y cero).' +
             ' Si la relación es positiva indica que si una de las características aumenta su valor, la otra también. Si la relación e snegativa indica lo contrario.')
    fig = px.imshow(round(st.session_state['datos'].select_dtypes(exclude='object').corr(),3), text_auto=True, aspect='auto')
    st.plotly_chart(fig,use_container_width=True)
    
    st.divider()
    st.subheader('Histograma')
    y_hist = st.selectbox('Histograma: ',st.session_state['datos'].columns, index= 0)
    hist_fig = px.histogram(st.session_state['datos'], x=y_hist)
    st.plotly_chart(hist_fig,use_container_width=True)

    st.divider()
    st.subheader('Scatter plot')
    col1, col2 = st.columns([1,1])
    with col1:
        x = st.selectbox('Eje horizontal scatter plot: ',st.session_state['datos'].columns, index=None)
    with col2:
        y = st.selectbox('Eje vertical scatter plot: ',st.session_state['datos'].columns, index=None)
    if((x is not None) & (y is not None)):
        df_temp = st.session_state['datos'].groupby([x], as_index=False).sum()
        fig = px.scatter(df_temp, x=x, y=y, title="Test scatter plot")
        st.plotly_chart(fig,use_container_width=True)
    
    del nan_
    del cero_
    del neg_
    del dup_
    del ac
    del x
    gc.collect()

