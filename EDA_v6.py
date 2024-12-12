import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA,HoltWinters, ARIMA, AutoETS, AutoRegressive, AutoCES
import os
from statsmodels.tsa.seasonal import seasonal_decompose 
from utilsforecast.losses import mae,mse,rmse,mape,smape
from utilsforecast.evaluation import evaluate
os.environ['NIXTLA_ID_AS_COL'] = '1'
pd.options.display.max_columns = None
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import numpy as np
from scipy import stats 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
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
#logo_path = "C:\\Users\\Celula1\\.streamlit\\img\\logo_small.png"
#icon = "C:\\Users\\Celula1\\.streamlit\\img\\icon.png"
st.set_page_config(page_title='DEMO pronósticos')

with st.sidebar:
    #st.logo(image=logo_path, link='https://datlas.mx/', size='large', icon_image=logo_path)
    st.title('EDA')
    with st.expander('Cargar Datos', expanded=True):
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
#   ESTADISTICA BASICA
if _file is not None:    
    st.header('Datos')
    st.write(st.session_state['datos'])
    st.header('Renglones y columnas')
    st.text('RENGLONES: ' + str(st.session_state['datos'].shape[0]))
    st.text('COLUMNAS: ' + str(st.session_state['datos'].shape[1])) 
    st.text(st.session_state['datos'].nunique())
    nan_ = c_nan(st.session_state['datos'])
    cero_=  c_cero(st.session_state['datos'])
    neg_=  c_negative(st.session_state['datos'])
    dup_=  c_duplicate(st.session_state['datos'])
    #st.write(dup_)
    info = pd.DataFrame(st.session_state['datos'].dtypes)
    info.columns = ['Tipo de dato']
    
    ac = pd.concat([cero_,neg_,nan_,info], axis=1)
    ac.style.format_index(str.upper, axis = 1)
    st.header('Ceros, negativos, Nan y tipos de dato')
    st.write(ac)
    st.header('Duplicados por columna')
    st.write(dup_)
    #st.write(st.session_state['datos'].describe().apply(lambda s: s.apply('{0:.0f}'.format)))
    st.header('Estadística básica')
    st.write(st.session_state['datos'].describe())

    st.header('Bar plot')
    col1, col2 = st.columns([1,1])
    with col1:
        x_bar = st.selectbox('Eje horizontal bar plot: ',st.session_state['datos'].columns, index=None)
    with col2:
        y_bar = st.selectbox('Eje vertical bar plot: ',st.session_state['datos'].columns, index=None)
    if((x_bar is not None) & (y_bar is not None)):
        df_temp = st.session_state['datos'].groupby([x_bar], as_index=False).sum()
        fig = px.bar(df_temp, x=x_bar, y=y_bar, title="Test bar plot")
        st.plotly_chart(fig,use_container_width=True)
    
    st.header('Line plot')
    col1, col2 = st.columns([1,1])
    with col1:
        x_line = st.selectbox('Eje horizontal line plot: ',st.session_state['datos'].columns, index=None)
    with col2:
        y_line = st.selectbox('Eje vertical line plot: ',st.session_state['datos'].columns, index=None)
    if((x_line is not None) & (y_line is not None)):
        df_temp = st.session_state['datos'].groupby([x_line], as_index=False).sum()
        fig = px.line(df_temp, x=x_line, y=y_line, title="Test line plot")
        st.plotly_chart(fig,use_container_width=True)

    st.header('Correlación')
    plt.figure(figsize=(8, 4))
    sns.heatmap(st.session_state['datos'].select_dtypes(exclude='object').corr(), cmap='viridis', annot = True)
    plt.title('Heatmap Correlación')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    st.header('Histograma')
    y_hist = st.selectbox('Histograma: ',st.session_state['datos'].columns, index= 0)
    hist_fig = px.histogram(st.session_state['datos'], x=y_hist)
    st.plotly_chart(hist_fig,use_container_width=True)

    st.header('Scatter plot')
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
    
#   -----------------------------------------------------------------------
#   MODIFICAR DATOS
if _file is not None:
    with st.sidebar:
        with st.expander('Modificar datos'):
            date = st.checkbox('Agregar columna de fecha', False)
            group = st.checkbox('Agrupar datos', False)
            imputation = st.checkbox('Imputar datos', False, disabled=True)
            outliers = st.checkbox('Modificar Outliers', False, disabled=True)
            
    if(date):
        _day_col = st.selectbox("Día:",st.session_state['datos'].select_dtypes(exclude=['object']).columns, index=None)
        _month_col = st.selectbox("Mes:",st.session_state['datos'].select_dtypes(exclude=['object']).columns, index=None)
        _year_col = st.selectbox("Año:",st.session_state['datos'].select_dtypes(exclude=['object']).columns, index=None)
        _week_col = st.selectbox("Semana:",st.session_state['datos'].select_dtypes(exclude=['object']).columns, index=None)

        if(_year_col is not None and _week_col is not None):
            #st.write(pd.to_datetime((st.session_state['datos'][_year_col].astype('int')*100 + st.session_state['datos'][_week_col].astype('int')).astype(str) + '0', format='%Y%W%w'))
            #st.session_state['datos']['FECHA'] = pd.to_datetime((st.session_state['datos'][_year_col].astype('int')*100 + st.session_state['datos'][_week_col].astype('int')).astype(str) + '0', format='%Y%W%w')
            temp = st.session_state['datos']
            temp['FECHA'] = pd.to_datetime((temp[_year_col].astype('int')*100 + temp[_week_col].astype('int')).astype(str) + '0', format='%Y%W%w')
            st.session_state['datos'] = temp
            st.write(st.session_state['datos'])
        if(_year_col is not None and _month_col is not None):
            print()
        if(_year_col is not None and _month_col is not None and _day_col is not None):
            print()
    if(group):
        selected_col = st.multiselect('Agrupar por las columnas:', st.session_state['datos'].columns,default=None)
        if(len(selected_col) != 0):
            st.write(st.session_state['datos'].groupby(selected_col, as_index = False).sum())
            _keep_df = st.checkbox("Conservar agrupación:", False)
            if(_keep_df):
                st.session_state['datos'] = st.session_state['datos'].groupby(selected_col, as_index = False).sum()
                st.write(st.session_state['datos'])
    if(imputation):
        st.write('Hacer algo para imputar')
    if(outliers):
        st.write('Hacer algo con outliers')
    gc.collect()                      
#   -----------------------------------------------------------------------
#   FORECAST
if _file is not None:
    st.header('Datos para pronóstico')
    x = st.selectbox('Columna de ID: ',st.session_state['datos'].columns, index=None)
    y = st.selectbox('Columna de tiempo: ',st.session_state['datos'].columns, index=None)
    z = st.selectbox('Columna de variable a pronosticar: ',st.session_state['datos'].columns, index=None)
    
    if((y is not None) & (z is not None)):
        df_forecast = st.session_state['datos'][[x,y,z]]
        df_forecast.columns = ['unique_id','ds','y']
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], dayfirst=True)
        st.header('Visualizar serie')
        st.write(df_forecast)
        fig = px.line(df_forecast, x = 'ds', y = 'y', title="Serie de tiempo", height=350)
        fig.update_traces(line={'width': .8, 'color': '#657695'})
        fig.update_xaxes(tickangle=90, dtick="M1")
        fig.update_layout(plot_bgcolor='#ebeff6')
        st.plotly_chart(fig,use_container_width=True)    
    
        st.header('Descomponer serie')
        stl = STL(df_forecast['y'], len(df_forecast['y']))
        result = stl.fit()
        plot_components(result)
    
    
        st.header('Outliers')
        o = identify_outliers(df_forecast, 'zscore')
        #o = identify_outliers(df_forecast, 'iqr')
        st.write(o)

        # Create line plot for titration curve
        fig = px.line(df_forecast, x='ds', y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},title= 'Outliers' ,height=350)
        fig.update_traces(line={'width': .8, 'color': '#657695'})
        fig.update_xaxes(tickangle=90, dtick="M1")
        fig.update_layout(plot_bgcolor='#ebeff6')
        # Add scatter plots for each set of equivalence points
        for i in range(len(o)):
            fig.add_trace(px.scatter(o.iloc[[i]], x='ds', y='y' , color_discrete_sequence=['red']).data[0])
        
        st.plotly_chart(fig,use_container_width=True)

        st.header('Pronóstico')
        season_l = 52 # week
        models = [AutoARIMA(season_length=season_l), HoltWinters(season_length=season_l)]
        sf = StatsForecast(models=models , freq='W', n_jobs = -1)
        df_test = df_forecast.tail(4)
        p = forecast(df_forecast.iloc[:-4], sf, 4)
        st.write(p.merge(df_test, on=['ds','unique_id']))
        plot_forecast(df_forecast,p)

        del x
        del y
        del z
        del models
        del sf
        del o
        gc.collect()