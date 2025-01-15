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
    df = df.rename(columns={0:'serie temporal','season':'Estacionalidad','trend':'Tendencia', 'resid':'Residuos'})
    components = df.columns
    rows = len(components)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles = [i for i in components])
  
    # Plot original data
    for i, col in enumerate(components):
        if(i == 1):
            fig.add_trace(go.Scatter(x=df.index, y=round(df[col],4), mode='lines', name=col), row=i+1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=i+1, col=1)

    # Update layout
    fig.update_layout(
        height=700,
    #    width=1200
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

def plot_forecast(db, pred):
    fig = px.line(db, x="ds", y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig.update_traces(line={'width': .8, 'color': '#000000'})
    fig.update_xaxes(tickangle=90, dtick="M1")
    fig.update_layout(plot_bgcolor='#ebeff6')
    # Add scatter plots for each set of equivalence points
    #for i in range(len(pred)):
        #fig.add_trace(px.line(pred.iloc[[i]], x="ds", y="AutoARIMA" , color_discrete_sequence=['red'], labels = 'AutoARIMA').data[0])
        #fig.add_trace(px.line(pred.iloc[[i]], x="ds", y="HoltWinters" , color_discrete_sequence=['yellow'], labels = 'AutoARIMA2').data[0])
        #fig.add_trace(px.line(pred.iloc[[i]], x="ds", y="AutoETS" , color_discrete_sequence=['blue'], labels = 'AutoARIMA3').data[0])
        #fig.add_trace(px.line(pred.iloc[[i]], x="ds", y="CES" , color_discrete_sequence=['green'], labels = 'AutoARIMA4').data[0])
        #fig.add_trace(px.line(pred.iloc[[i]], x="ds", y="AutoRegressive" , color_discrete_sequence=['black'],labels = 'AutoARIMA5').data[0])

    fig.add_trace(px.line(pred, x="ds", y="AutoARIMA" , color_discrete_sequence=['#f57c74'], width=11).data[0])
    fig.add_trace(px.line(pred, x="ds", y="HoltWinters" , color_discrete_sequence=['#f5de74']).data[0])
    fig.add_trace(px.line(pred, x="ds", y="AutoETS" , color_discrete_sequence=['#748bf5']).data[0])
    fig.add_trace(px.line(pred, x="ds", y="CES" , color_discrete_sequence=['#6ccc66']).data[0])
    fig.add_trace(px.line(pred, x="ds", y="AutoRegressive" , color_discrete_sequence=['gray']).data[0])
    
    def selector(column_name):
        # just need to be careful that "column_name" is not any other string in "hovertemplate" data
        f = lambda x: True if column_name in x['hovertemplate'] else False
        return f
    
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("AutoARIMA"))
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("AutoETS"))
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("HoltWinters"))
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("CES"))
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("AutoRegressive"))
    st.plotly_chart(fig,use_container_width=True)

#   -----------------------------------------------------------------------
#   LOCAL
#logo_path = "C:/Users/Celula1/app/static/logo_small.png"
#icon = "C:/Users/Celula1/app/static/icon.png"
#img_forecast = "C:\\Users\\Celula1\\app\\static\\dg_time_series2.png"

#   PLOOMBER
logo_path = "static/logo_small.png"
icon = "static/icon.png"
img_forecast = "static/dg_time_series2.png"


with st.sidebar:
    st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.header('Pasos para generar un pronóstico de series de tiempo')
st.image(img_forecast, caption='En la imágen vemos los pasos que se deberían seguir para generar un pronóstico de series de tiempo, para este DEMO se omiten algunos.')

st.header('Archivos ejemplo para pronósticos')
_dataset1 = st.checkbox("Pasajeros de aerolinea", False)
_dataset2 = st.checkbox("Producción de electricidad", False, disabled=True)
_dataset3 = st.checkbox("Ventas", False, disabled=True)

if(_dataset1):
    #st.session_state['respaldo'] = read_file('C:/Users/Celula1/app/Forecast/airline_passengers.csv')
    st.session_state['respaldo'] = read_file('Forecast/airline_passengers.csv')
    st.session_state['respaldo']['MONTH'] = pd.to_datetime(st.session_state['respaldo']['MONTH'])
    st.session_state['datos'] = st.session_state['respaldo']
    st.subheader('Visualización de datos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.write(st.session_state['respaldo'])
if(_dataset2):
    #st.session_state['respaldo'] = read_file('C:/Users/Celula1/app/Forecast/ETTh1.csv')
    st.session_state['respaldo'] = read_file('Forecast/ETTh1.csv')
    st.session_state['datos'] = st.session_state['respaldo']
    st.write(st.session_state['datos'])
if(_dataset3):
    #st.session_state['respaldo'] = read_file('C:/Users/Celula1/app/Forecast/airline_passengers.csv')
    st.session_state['respaldo'] = read_file('Forecast/airline_passengers.csv')
    st.session_state['datos'] = st.session_state['respaldo']
    st.write(st.session_state['datos'])
    
gc.collect()
                 
#   -----------------------------------------------------------------------
#   FORECAST
if 'datos' in st.session_state:
    df_forecast = st.session_state['respaldo']
    df_forecast.columns = ['unique_id','ds','y']
    
    #   -----------------------------------------------------------------------------------------------
    #st.subheader('Serie de tiempo')
    #st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    #fig = px.line(df_forecast, x = 'ds', y = 'y', height=350)
    #fig.update_traces(line={'width': .8, 'color': '#657695'})
    #fig.update_xaxes(tickangle=90, dtick="M1")
    #fig.update_layout(plot_bgcolor='#ebeff6')
    #st.plotly_chart(fig,use_container_width=True)    
    
    #   -----------------------------------------------------------------------------------------------
    stl = STL(df_forecast['y'], len(df_forecast['y']))
    result = stl.fit()
    st.subheader('Descomposición de la serie')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    plot_components(result)

    #   -----------------------------------------------------------------------------------------------
    st.subheader('Valores atípicos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    o = identify_outliers(df_forecast, 'zscore')
    #o = identify_outliers(df_forecast, 'iqr')
    st.write(o)

    # Create line plot for titration curve
    fig = px.line(df_forecast, x="ds", y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig.update_traces(line={'width': .8, 'color': '#657695'})
    fig.update_xaxes(tickangle=90, dtick="M1")
    fig.update_layout(plot_bgcolor='#ebeff6')
    # Add scatter plots for each set of equivalence points
    for i in range(len(o)):
        fig.add_trace(px.scatter(o.iloc[[i]], x="ds", y="y" , color_discrete_sequence=['red']).data[0])
    
    st.plotly_chart(fig,use_container_width=True)

    #   -----------------------------------------------------------------------------------------------
    
    if(_dataset2):  #   ELECTRICIDAD
        season_l = 365 
        models = [HoltWinters(season_length=season_l)]
        sf = StatsForecast(models=models , freq='D', n_jobs = -1)
        p = sf.forecast(df= df_forecast, h= 4, fitted=True)
        st.write(p)
        plot_forecast(df_forecast,p)
    if(_dataset3):  #   VENTAS
        season_l = 52
        models = [HoltWinters(season_length=season_l)]
        sf = StatsForecast(models=models , freq='W', n_jobs = -1)
        p = sf.forecast(df= df_forecast, h= 4, fitted=True)
        st.write(p)
        plot_forecast(df_forecast,p)
    if(_dataset1):  #   AEROLINEA
        season_l = 12
        #models = [HoltWinters(season_length=season_l)]
        models = [AutoARIMA(season_length=season_l), 
                  HoltWinters(season_length=season_l),
                  AutoETS(season_length=season_l),
                  AutoCES(season_length=season_l),
                  AutoRegressive(lags=14)]
    
        st.subheader('Pronóstico')
        st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
        sf = StatsForecast(models=models , freq='M', n_jobs = -1)
        p = sf.forecast(df= df_forecast[:-4], h= 4, fitted=True)
        p['y'] = list(df_forecast.tail(4)['y'])
        st.write(p)
        plot_forecast(df_forecast,p)

    del df_forecast
    del st.session_state['datos']
    del st.session_state['respaldo']
    gc.collect()

