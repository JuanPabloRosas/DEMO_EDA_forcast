import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA,HoltWinters, ARIMA, AutoETS, AutoRegressive, AutoCES
import os
from utilsforecast.losses import mae,mse,rmse,mape,smape
from utilsforecast.evaluation import evaluate
os.environ['NIXTLA_ID_AS_COL'] = '1'
pd.options.display.max_columns = None
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import numpy as np
from scipy import stats
import gc

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

#   -----------------------------------------------------------------------
def read_file(filename):
    try:
        db = pd.read_csv(filename, encoding='utf-8', thousands=',')
    except:
        try:
            db = pd.read_csv(filename, encoding='utf-8-sig', thousands=',')
        except:
            db = pd.read_csv(filename, encoding='latin1', thousands=',')
    
    db.columns = db.columns.str.upper()
    return db

def plot_components(result):
    df = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
    df = df.rename(columns={0:'Original Data', 'season':'seasonal','observed':'Original Data'})
    components = df.columns
    rows = len(components)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles = [i for i in components])
  
    # Plot original data
    for i, col in enumerate(components):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=i+1, col=1)

    # Update layout
    fig.update_layout(
        title='Time Series Decomposition',
        xaxis_title='Time',
        height=1200,
        width=1200
    )
    st.plotly_chart(fig,use_container_width=True)

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

def forecast(db_forecast, sf, horiz):
    db_forecast.set_index(db_forecast['ds'], inplace=True)
    db_forecast = db_forecast.asfreq('W')
    db_forecast.interpolate(method='bfill', inplace=True)
    pred = sf.forecast(df= db_forecast, h= horiz) 
    return pred

def plot_forecast(db, pred):
    fig = px.line(db, x="ds", y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},title= 'Forecast' ,height=350)
    fig.update_traces(line={'width': .8, 'color': '#657695'})
    fig.update_xaxes(tickangle=90, dtick="M1")
    fig.update_layout(plot_bgcolor='#ebeff6')
    # Add scatter plots for each set of equivalence points
    for i in range(len(pred)):
        fig.add_trace(px.scatter(pred.iloc[[i]], x="ds", y="AutoARIMA" , color_discrete_sequence=['red']).data[0])
        fig.add_trace(px.scatter(pred.iloc[[i]], x="ds", y="HoltWinters" , color_discrete_sequence=['yellow']).data[0])
    st.plotly_chart(fig,use_container_width=True)

#   -----------------------------------------------------------------------
logo_path = "C:\\Users\\Celula1\\.streamlit\\img\\logo_small.png"
icon = "C:\\Users\\Celula1\\.streamlit\\img\\icon.png"

#st.header('DEMO Forecast')
#st.markdown('---')
#st.markdown("""
#    <div style='background-color: #f0f2f5; padding: 10px; border-radius: 5px;'>
#        <h2 style='color: #333;'>Este es un contenido dentro de un div simulado</h2>
#        <p>Aquí puedes agregar cualquier información.</p>
#    </div>
#""", unsafe_allow_html=True)

st.title('Archivo para pronósticos')
with st.expander('Cargar datos',expanded=True):
    _dataset = st.checkbox("Dataset precargado", False, disabled=True)
    if(_dataset):
        st.selectbox("Selecciona el dataset",options=[1,2,3])
    else:
        _file = st.file_uploader("Carga un archivo CSV", type="csv")
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
#   FORECAST
if _file is not None:
    #_desc_serie = st.checkbox("Descomponer Serie", False)
    #_see = st.checkbox("Visualizar Serie", False)
    #_imputar = st.checkbox("Imputar", False)
    #_outliers = st.checkbox("Outliers", False)
    #_transformar = st.checkbox("Transformar", False)
    #_vc = st.checkbox("Validación cruzada", False)
    #_pred = st.checkbox("Predecir", False)
    
    col1, col2, col3 = st.columns([1,1,1])
    x_values, y_values = None, None
    with col1:
        x = st.selectbox('Columna de ID: ',st.session_state['datos'].columns, index=None)
    with col2:
        y = st.selectbox('Columna de tiempo: ',st.session_state['datos'].columns, index=None)
    with col3:
        z = st.selectbox('Columna de variable a pronosticar: ',st.session_state['datos'].columns, index=None)
    
    if((x is not None) & (y is not None) & (z is not None)):
        st.session_state['forecast'] = st.session_state['datos'].groupby([x,y], as_index = False)[x,y,z].sum()
        df_forecast = st.session_state['forecast']
        df_forecast = df_forecast[[x,y,z]]
        df_forecast.columns = ['unique_id','ds','y']
        st.write(df_forecast)
        
        fig = px.line(df_forecast, x = 'ds', y = 'y', title="Serie de tiempo", height=350)
        fig.update_traces(line={'width': .8, 'color': '#657695'})
        fig.update_xaxes(tickangle=90, dtick="M1")
        fig.update_layout(plot_bgcolor='#ebeff6')
        st.plotly_chart(fig,use_container_width=True)    
        if('forecast' in st.session_state):
            stl = STL(df_forecast['y'], len(df_forecast['y']))
            result = stl.fit()
            plot_components(result)
        
            o = identify_outliers(df_forecast, 'zscore')
            #o = identify_outliers(df_forecast, 'iqr')
            st.write(o)

            # Create line plot for titration curve
            fig = px.line(df_forecast, x="ds", y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},title= 'Outliers' ,height=350)
            fig.update_traces(line={'width': .8, 'color': '#657695'})
            fig.update_xaxes(tickangle=90, dtick="M1")
            fig.update_layout(plot_bgcolor='#ebeff6')
            # Add scatter plots for each set of equivalence points
            for i in range(len(o)):
                fig.add_trace(px.scatter(o.iloc[[i]], x="ds", y="y" , color_discrete_sequence=['red']).data[0])
            
            st.plotly_chart(fig,use_container_width=True)

            #fig = px.line(o, x="ds", y='y', width=750, labels={'ds': 'time(ds)', 'y': 'target(y)'})
            #st.plotly_chart(fig,use_container_width=True)
        
            season_l = 52 # week
            #models = [AutoARIMA(season_length=season_l), 
            #          HoltWinters(season_length=season_l),
            #          AutoETS(season_length=season_l),
            #          AutoCES(season_length=season_l),
            #          AutoRegressive(lags=14)]
            models = [AutoARIMA(season_length=season_l), HoltWinters(season_length=season_l)]
            sf = StatsForecast(models=models , freq='W', n_jobs = -1)
            p = forecast(df_forecast, sf, 4)
            st.write(p)
            plot_forecast(df_forecast,p)
