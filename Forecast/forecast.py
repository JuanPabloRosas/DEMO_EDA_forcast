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
logo_path = "./app/static/logo_small.png"
icon = "./app/static/icon.png"

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.header('Pasos para generar un pronóstico de series de tiempo')
st.image("./app/static/dg_time_series2.png", caption='En la imágen vemos los pasos que se deberían seguir para generar un pronóstico de series de tiempo, para este DEMO se omiten algunos.')

st.header('Archivos ejemplo para pronósticos')
_dataset1 = st.checkbox("Pasajeros de aerolinea", False)
_dataset2 = st.checkbox("Producción de electricidad", False)
_dataset3 = st.checkbox("Ventas", False, disabled=True)

if(_dataset1):
    st.session_state['respaldo'] = read_file('./app/Forecast/airline_passengers.csv')
    st.session_state['respaldo']['MONTH'] = pd.to_datetime(st.session_state['respaldo']['MONTH'])
    st.session_state['datos'] = st.session_state['respaldo']
    st.write(st.session_state['respaldo'])
if(_dataset2):
    st.session_state['respaldo'] = read_file('./app/Forecast/ETTh1.csv')
    st.session_state['datos'] = st.session_state['respaldo']
    st.write(st.session_state['datos'])
if(_dataset3):
    st.session_state['respaldo'] = read_file('./app/Forecast/airline_passengers.csv')
    st.session_state['datos'] = st.session_state['respaldo']
    st.write(st.session_state['datos'])
    
gc.collect()
                 
#   -----------------------------------------------------------------------
#   FORECAST
if 'datos' in st.session_state:
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
        
        #   -----------------------------------------------------------------------------------------------
        fig = px.line(df_forecast, x = 'ds', y = 'y', title="Serie de tiempo", height=350)
        fig.update_traces(line={'width': .8, 'color': '#657695'})
        fig.update_xaxes(tickangle=90, dtick="M1")
        fig.update_layout(plot_bgcolor='#ebeff6')
        st.plotly_chart(fig,use_container_width=True)    
        
        #   -----------------------------------------------------------------------------------------------
        stl = STL(df_forecast['y'], len(df_forecast['y']))
        result = stl.fit()
        plot_components(result)
    
        #   -----------------------------------------------------------------------------------------------
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

        #   -----------------------------------------------------------------------------------------------
        #models = [AutoARIMA(season_length=season_l), 
            #          HoltWinters(season_length=season_l),
            #          AutoETS(season_length=season_l),
            #          AutoCES(season_length=season_l),
            #          AutoRegressive(lags=14)]
        
        tiempo = st.selectbox('Pronóstico por: ',['Día', 'Semana', 'Mes'], index=None)
        if(tiempo == 'Día'):
            season_l = 365
            models = [HoltWinters(season_length=season_l)]
            sf = StatsForecast(models=models , freq='D', n_jobs = -1)
            p = forecast(df_forecast, sf, 4)
            st.write(p)
            plot_forecast(df_forecast,p)
        if(tiempo == 'Semana'):
            season_l = 52
            models = [HoltWinters(season_length=season_l)]
            sf = StatsForecast(models=models , freq='W', n_jobs = -1)
            p = forecast(df_forecast, sf, 4)
            st.write(p)
            plot_forecast(df_forecast,p)
        if(tiempo == 'Mes'):
            season_l = 12
            models = [HoltWinters(season_length=season_l)]
            sf = StatsForecast(models=models , freq='M', n_jobs = -1)
            p = forecast(df_forecast, sf, 4)
            st.write(p)
            plot_forecast(df_forecast,p)

del df_forecast
del st.session_state['forecast']
del st.session_state['datos']
del st.session_state['respaldo']
gc.collect()
            
            
