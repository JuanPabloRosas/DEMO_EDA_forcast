import streamlit as st
import pandas as pd 
import plotly.express as px
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA,HoltWinters, AutoETS, AutoRegressive, AutoCES
import os
from utilsforecast.losses import *
from utilsforecast.evaluation import evaluate
os.environ['NIXTLA_ID_AS_COL'] = '1'
pd.options.display.max_columns = None
from statsmodels.tsa.seasonal import STL
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
    #fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles = [i for i in components])
  
    # Plot original data
    #for i, col in enumerate(components):
    #    if(i == 1):
    #        fig.add_trace(go.Scatter(x=df.index, y=round(df[col],4), mode='lines', name=col), row=i+1, col=1)
    #    else:
    #        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=i+1, col=1)

    fig1 = px.line(df, x=df.index, y='Tendencia', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig1.update_traces(line={'width': .8, 'color': '#657695'})
    fig1.update_xaxes(tickangle=90, dtick="M1")
    fig1.update_layout(plot_bgcolor='#ebeff6')
    st.plotly_chart(fig1,use_container_width=True)

    fig2 = px.line(df, x=df.index, y='Estacionalidad', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig2.update_traces(line={'width': .8, 'color': '#657695'})
    fig2.update_xaxes(tickangle=90, dtick="M1")
    fig2.update_layout(plot_bgcolor='#ebeff6')
    st.plotly_chart(fig2,use_container_width=True)

    fig3 = px.line(df, x=df.index, y='Residuos', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig3.update_traces(line={'width': .8, 'color': '#657695'})
    fig3.update_xaxes(tickangle=90, dtick="M1")
    fig3.update_layout(plot_bgcolor='#ebeff6')
    st.plotly_chart(fig3,use_container_width=True)

    #-------------------------------
    # Paso 1: Gráfico de residuos vs. tiempo (residual plot)
    #residual_plot = go.Figure()
    #residual_plot.add_trace(go.Scatter(df,x=df.index, y='Residuos',mode='markers+lines',name='Residuos',line=dict(color='blue'),marker=dict(color='blue')))
    #residual_plot.add_trace(go.Scatter(df,x=df.index,y='Residuos',mode='lines',name='Cero',line=dict(color='red', dash='dash')))
    #residual_plot.update_layout(title='Gráfico de Residuos vs. Tiempo',xaxis_title='Tiempo',yaxis_title='Residuo',template='plotly_dark')

    # Paso 2: Histograma de residuos
    histograma_residuos = px.histogram(df['Residuos'], nbins=20, title="Histograma de Residuos", labels={'value': 'Residuo'})
    histograma_residuos.update_traces(marker_color='green', opacity=0.75)
    histograma_residuos.update_layout(
        xaxis_title='Residuo',
        yaxis_title='Frecuencia',
        template='plotly_dark'
    )

    # Paso 3: Gráfico Q-Q (Comparación con la distribución normal)
    # Generamos el gráfico Q-Q usando statsmodels y luego lo pasamos a Plotly

    # Q-Q plot
    #qq_data = sm.qqplot(df['Residuos'], line ='45', plot=None)
    #qq_x = qq_data.axes[0].get_lines()[0].get_xdata()
    #qq_y = qq_data.axes[0].get_lines()[0].get_ydata()

    #qq_plot = go.Figure()

    #qq_plot.add_trace(go.Scatter(    x=qq_x, y=qq_y,mode='markers',name='Puntos Q-Q',marker=dict(color='purple')))

    #qq_plot.update_layout(    title='Gráfico Q-Q de Residuos',xaxis_title='Cuantiles Teóricos',yaxis_title='Cuantiles Muestrales',template='plotly_dark')

    # Mostrar los gráficos
    #st.plotly_chart(residual_plot,use_container_width=True)
    st.plotly_chart(histograma_residuos,use_container_width=True)
    #st.plotly_chart(qq_plot,use_container_width=True)
    #-------------------------------

def identify_outliers(db, m):
    if(m == 'zscore'):
        z = np.abs(stats.zscore( db['y']))
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

def plot_forecast(db, pred,modelos):
    fig = px.line(db, x="ds", y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig.update_traces(line={'width': .8, 'color': '#000000'})
    fig.update_xaxes(tickangle=90, dtick="M1")
    fig.update_layout(plot_bgcolor='#ebeff6')
    
    #fig.add_trace(px.line(pred, x="ds", y="AutoARIMA" , color_discrete_sequence=['#f57c74'], width=11).data[0])
    #fig.add_trace(px.line(pred, x="ds", y="HoltWinters" , color_discrete_sequence=['#f5de74']).data[0])
    #fig.add_trace(px.line(pred, x="ds", y="AutoETS" , color_discrete_sequence=['#748bf5']).data[0])
    #fig.add_trace(px.line(pred, x="ds", y="CES" , color_discrete_sequence=['#6ccc66']).data[0])
    #fig.add_trace(px.line(pred, x="ds", y="AutoRegressive" , color_discrete_sequence=['gray']).data[0])
    
    def selector(column_name):
        f = lambda x: True if column_name in x['hovertemplate'] else False
        return f
    for m in modelos:
        fig.add_trace(px.line(pred, x="ds", y=m , color_discrete_sequence=['#f57c74'], width=11).data[0])
        fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector(m))
    
    #fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("AutoARIMA"))
    #fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("AutoETS"))
    #fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("HoltWinters"))
    #fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("CES"))
    #fig.update_traces(patch={"line": {"dash": "dot"}}, selector=selector("AutoRegressive"))
    
    st.plotly_chart(fig,use_container_width=True)

#   -----------------------------------------------------------------------

#   DOCKER
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
#_aerolinea = st.checkbox("Pasajeros de aerolinea", False)
#_ventas = st.checkbox("Ventas", False)

dataset = st.radio(
    "Selecciona un dataset de prueba",
    ["Datos de aerolinea", "Datos de ventas"],
    captions=[
        "Datos de fechas de vuelos y cantidad de pasajeros",
        "Datos de transacciones de diferentes productos",
    ],
)

if(dataset  == 'Datos de aerolinea'):  #   AEROLINEA
    st.session_state['respaldo'] = read_file('Forecast/airline_passengers.csv')
    st.session_state['respaldo']['MONTH'] = pd.to_datetime(st.session_state['respaldo']['MONTH'])
    st.session_state['datos'] = st.session_state['respaldo']
    st.subheader('Visualización de datos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.write(st.session_state['respaldo'])
if(dataset == 'Datos de ventas'):  #   VENTAS
    st.session_state['respaldo'] = read_file('Forecast/sales_data_sample2.csv')
    st.session_state['datos'] = st.session_state['respaldo']
    st.subheader('Visualización de datos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.write(st.session_state['respaldo'])
    
                 
#   -----------------------------------------------------------------------
#   FORECAST
if(dataset == 'Datos de ventas'):  #   VENTAS
    df_forecast = st.session_state['respaldo']
    df_forecast = df_forecast[['PRODUCTO','CANTIDAD','FECHA']]
    df_forecast['mes'] = pd.to_datetime(df_forecast['FECHA'], dayfirst = True).dt.month
    df_forecast['year'] = pd.to_datetime(df_forecast['FECHA'], dayfirst=True).dt.year
    df_forecast.columns = ['unique_id','y','ds','mes','year']
    
    df_forecast = df_forecast.groupby(['unique_id','mes','year'], as_index=False).sum()
    df_forecast['ds'] = pd.to_datetime(df_forecast['year'].astype(str) + df_forecast['mes'].astype(str), format='%Y%m')
    df_forecast = df_forecast[['unique_id','ds','y']]
    df_forecast = df_forecast.sort_values('ds')

    st.subheader('Descomposición de la serie')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.markdown("""
        Los datos de :blue-background[**series de tiempo**] pueden mostrar una gran variedad de patrones, y a menudo 
        resulta útil dividir una serie temporal en varios componentes, cada uno de los cuales representa una categoría de patrón subyacente.
        
        Cuando descomponemos una serie temporal en componentes, solemos combinar la tendencia y el ciclo en un único componente de tendencia-ciclo 
        (a menudo denominado simplemente tendencia para simplificar). 
        Así pues, podemos considerar que una serie temporal consta de tres componentes: un componente de tendencia-ciclo, un componente estacional y un 
        componente restante (que contiene cualquier otro elemento de la serie temporal). En algunas series temporales (por ejemplo, las que se observan al 
        menos diariamente), puede haber más de un componente estacional, correspondiente a los distintos periodos estacionales.
        
        > - **Tendencia**: Cálculo de medidas como la media, mediana, desviación estándar, percentiles, etc.
        > - **Estacionalidad**: Usar gráficos como histogramas, diagramas de dispersión, diagramas de barras, etc., para identificar patrones, tendencias y posibles outliers.
        > - **Ciclos**: Identificar puntos de datos que se alejan de manera significativa del resto del conjunto.
        
        """)
    prod = st.selectbox('Selecciona un producto: ',df_forecast['unique_id'].unique(), index=0, placeholder='Selecciona una opción')
    db = df_forecast[df_forecast['unique_id'] == prod]
    db.index = db['ds']
    stl = STL(db['y'], len(db['y']))
    result = stl.fit()    
    plot_components(result)

    #   -----------------------------------------------------------------------------------------------
    st.subheader('Valores atípicos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.markdown("""
        Un outlier (o "valor atípico" en español) es un valor en un conjunto de datos que se encuentra significativamente alejado de los demás. 
                En otras palabras, es un dato que se desvía considerablemente del patrón general o la tendencia que sigue el resto de los datos. 
                Los outliers pueden ser el resultado de errores en la recolección de datos, variabilidad natural, o incluso fenómenos interesantes 
                que merecen ser investigados más a fondo.
        """)
    o = identify_outliers(db, 'zscore')
    #o = identify_outliers(df_forecast, 'iqr')
    st.write(o)

    # Create line plot for titration curve
    fig = px.line(db, x=pd.to_datetime(db['ds']), y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig.update_traces(line={'width': .8, 'color': '#657695'})
    fig.update_xaxes(tickangle=90, dtick="M1")
    fig.update_layout(plot_bgcolor='#ebeff6')
    
    # Add scatter plots for each set of equivalence points
    for i in range(len(o)):
        fig.add_trace(px.scatter(o.iloc[[i]], x="ds", y="y" , color_discrete_sequence=['red']).data[0])
    
    st.plotly_chart(fig,use_container_width=True)

if(dataset  == 'Datos de aerolinea'):
    db = st.session_state['respaldo']
    db.columns = ['unique_id','ds','y']

    stl = STL(db['y'], len(db['y']))
    result = stl.fit()
    st.subheader('Descomposición de la serie')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.markdown("""
        Los datos de :blue-background[**series de tiempo**] pueden mostrar una gran variedad de patrones, y a menudo 
        resulta útil dividir una serie temporal en varios componentes, cada uno de los cuales representa una categoría de patrón subyacente.
        
        Cuando descomponemos una serie temporal en componentes, solemos combinar la tendencia y el ciclo en un único componente de tendencia-ciclo 
        (a menudo denominado simplemente tendencia para simplificar). 
        Así pues, podemos considerar que una serie temporal consta de tres componentes: un componente de tendencia-ciclo, un componente estacional y un 
        componente restante (que contiene cualquier otro elemento de la serie temporal). En algunas series temporales (por ejemplo, las que se observan al 
        menos diariamente), puede haber más de un componente estacional, correspondiente a los distintos periodos estacionales.
        
        > - **Tendencia**: Existe una tendencia cuando hay un aumento o una disminución a largo plazo en los datos. No tiene por qué ser lineal. 
                A veces nos referiremos a una tendencia como «cambio de dirección», cuando puede pasar de una tendencia creciente a una tendencia decreciente.
        > - **Estacionalidad**: Un patrón estacional se produce cuando una serie temporal se ve afectada por factores estacionales como la época del año o el día de la semana. 
                La estacionalidad es siempre de un periodo fijo y conocido.
        > - **Ciclos**: Un ciclo se produce cuando los datos muestran subidas y bajadas que no tienen una frecuencia fija. 
                Estas fluctuaciones suelen deberse a las condiciones económicas, y a menudo se relacionan con el «ciclo económico».
        """)
    plot_components(result)

    #   -----------------------------------------------------------------------------------------------
    st.subheader('Valores atípicos')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.markdown("""
        Un outlier (o "valor atípico" en español) es un valor en un conjunto de datos que se encuentra significativamente alejado de los demás. 
                En otras palabras, es un dato que se desvía considerablemente del patrón general o la tendencia que sigue el resto de los datos. 
                Los outliers pueden ser el resultado de errores en la recolección de datos, variabilidad natural, o incluso fenómenos interesantes 
                que merecen ser investigados más a fondo.
        """)
    o = identify_outliers(db, 'zscore')
    #o = identify_outliers(df_forecast, 'iqr')
    st.write(o)

    # Create line plot for titration curve
    fig = px.line(db, x="ds", y='y', labels={'ds': 'time(ds)', 'y': 'target(y)'},height=350)
    fig.update_traces(line={'width': .8, 'color': '#657695'})
    fig.update_xaxes(tickangle=90, dtick="M1")
    fig.update_layout(plot_bgcolor='#ebeff6')
    # Add scatter plots for each set of equivalence points
    for i in range(len(o)):
        fig.add_trace(px.scatter(o.iloc[[i]], x="ds", y="y" , color_discrete_sequence=['red']).data[0])
    
    st.plotly_chart(fig,use_container_width=True)

if((dataset == 'Datos de ventas') | (dataset  == 'Datos de aerolinea')):
    st.subheader('Pronóstico')
    st.markdown("""<hr style=" color: #E8AC13; border: 5px solid; display: inline-block; width: 50%; margin: auto;" /> """, unsafe_allow_html=True)
    st.markdown("""
        El proceso de selección de un modelo de pronóstico para series de tiempo implica varias etapas clave, que incluyen el análisis de los datos, 
        la identificación de patrones, la evaluación de diferentes modelos y la validación de los resultados.
        Antes de seleccionar cualquier modelo, es importante analizar la serie de tiempo en detalle. Algunas preguntas clave a considerar son:

        > - **Tendencia**: ¿Los datos muestran una dirección general a largo plazo, como un crecimiento o disminución constante?
        > - **Estacionalidad:**: ¿Existen patrones que se repiten a intervalos regulares, como fluctuaciones anuales, mensuales o semanales?
        > - **Ciclicidad**: ¿Los datos siguen ciclos irregulares que no son estrictamente estacionales?
        > - **Ruido**: ¿Hay variabilidad o fluctuaciones aleatorias en los datos?
        """)
    season_l = 12
    if(dataset  == 'Datos de aerolinea'):
        modelos = ['AutoARIMA','HoltWinters','AutoETS','CES','AutoRegressive']
    else:
        modelos = ['AutoARIMA','AutoETS','CES','AutoRegressive']
    
    m = st.selectbox('Selecciona un modelo: ',modelos, index=None, placeholder='Selecciona una opción')
    if(m is not None):
        if(m == 'AutoARIMA'):
            models = [AutoARIMA(season_length=season_l)]
        elif(m =='HoltWinters'):
            models = [HoltWinters(season_length=season_l)]
        elif(m =='AutoETS'):
            models = [AutoETS(season_length=season_l)]
        elif(m == 'CES'):
            models = [AutoCES(season_length=season_l)]
        else:
            models = [AutoRegressive(lags=14)]


        sf = StatsForecast(models=models, freq='M', n_jobs = -1)
        p = sf.forecast(df= db[:-4], h= 4, fitted=True, level=[90])
        p['y'] = list(db.tail(4)['y'])
        st.write(p)
        plot_forecast(db,p, [m, m+'-lo-90', m+'-hi-90'])
        
        metrics = [mae,mse,rmse,mape,smape]
        evaluation = evaluate(p, metrics=metrics)
        st.markdown("""
        Una vez seleccionados los modelos posibles, es necesario evaluarlos para determinar cuál proporciona las mejores predicciones. 
        Esto se realiza utilizando una muestra de prueba o mediante técnicas de validación cruzada. Las métricas comunes para evaluar el rendimiento incluyen:
                
        > - **Error cuadrático medio (RMSE) o Error absoluto medio (MAE)**: Miden la magnitud del error entre las predicciones y los valores reales.
        > - **El Error Porcentual Absoluto Medio (MAPE)**: Es una métrica que mide el error en términos relativos, es decir, expresa el error como un porcentaje de los valores reales.
        > - **El Error Porcentual Absoluto Medio Simétrico (sMAPE)**: Es una variante del MAPE diseñada para ser más robusta, especialmente en situaciones donde los valores reales son cercanos a cero.
        """)
        st.write(evaluation)

#   -----------------------------------------------------------------------------------------------
    
if 'datos' in st.session_state:
    del st.session_state['datos']
if 'respaldo' in st.session_state:
    del st.session_state['respaldo']
gc.collect()


