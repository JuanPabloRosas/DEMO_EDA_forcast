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

image = "./app/static/bg3.avif"

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


#   -----------------------------------------------------------------------
#   MODIFICAR DATOS

st.title('Archivo para modificar')
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

if _file is not None:
    with st.sidebar:
        with st.expander('Modificar datos'):
            date = st.checkbox('Crear columna de fecha', False)
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
        
