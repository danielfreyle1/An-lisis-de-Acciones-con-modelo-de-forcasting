import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplcyberpunk
import numpy as np
from prophet import Prophet


plt.style.use("dark_background")
###########################
#### Funciones Principales
###########################

def get_data(stock, start_time, end_time):
    df = yf.download(
        stock, 
        start=start_time, 
        end=end_time, 
        auto_adjust=True, 
        group_by='column'   # 👈 evita que se creen MultiIndex
    )

    # Si las columnas siguen siendo MultiIndex, las aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = df.columns.str.lower()

    return df


def get_levels(dfvar):
    def isSupport(df, i):
        support = (
            df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i+1] and 
            df['low'].iloc[i+1] < df['low'].iloc[i+2] and 
            df['low'].iloc[i-1] < df['low'].iloc[i-2]
        )
        return support

    def isResistance(df, i):
        resistance = (
            df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i+1] and 
            df['high'].iloc[i+1] > df['high'].iloc[i+2] and 
            df['high'].iloc[i-1] > df['high'].iloc[i-2]
        )
        return resistance

    def isFarFromLevel(l, levels, s):
        level = np.sum([abs(l - x[1]) < s for x in levels])
        return level == 0

    df = dfvar.copy()

    # 🔑 Normaliza columnas (maneja MultiIndex y mayúsculas/minúsculas)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().lower() for col in df.columns.values]
    else:
        df.columns = df.columns.str.lower()

    s = np.mean(df['high'] - df['low'])
    levels = []
    for i in range(2, df.shape[0]-2):
        if isSupport(df, i):  
            levels.append((i, df['low'].iloc[i]))
        elif isResistance(df, i):
            levels.append((i, df['high'].iloc[i]))

    filter_levels = []
    for i in range(2, df.shape[0]-2):
        if isSupport(df, i):
            l = df['low'].iloc[i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i, l))
        elif isResistance(df, i):
            l = df['high'].iloc[i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i, l))

    return filter_levels

################################################

def plot_close_price(data):
    #  Normaliza columnas (por si vienen con mayúsculas o MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip().lower() for col in data.columns.values]
    else:
        data.columns = data.columns.str.lower()

    # Llamamos a get_levels    
    levels = get_levels(data)


    df_levels = pd.DataFrame(levels, columns=['index','close']) if isinstance(levels, list) else levels
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(data.index, data['close'], color='dodgerblue', linewidth=1)
    mplcyberpunk.add_glow_effects()

    if df_levels.empty:
        ax.text(0.5, 0.5, "No se encontraron niveles", transform=ax.transAxes,
                ha='center', va='center', color='white', fontsize=14)
        return fig  # 👈 devuelve fig aunque no haya niveles

    df_levels.set_index('index', inplace=True)
    max_level = df_levels['close'].idxmax()
    min_level = df_levels['close'].idxmin()
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

    if df_levels.loc[min_level, 'close'] > df_levels.loc[max_level, 'close']:
        fib_levels = [
            data['close'].iloc[max_level] - (data['close'].iloc[max_level] - data['close'].iloc[min_level]) * ratio
            for ratio in ratios
        ]
    else:
        fib_levels = [
            data['close'].iloc[min_level] + (data['close'].iloc[max_level] - data['close'].iloc[min_level]) * ratio
            for ratio in ratios
        ]

    for level, ratio in zip(fib_levels, ratios):
        ax.hlines(level, xmin=data.index[0], xmax=data.index[-1],
                  colors='snow', linestyles='dotted', linewidth=0.9,
                  label=f"{ratio*100:.1f}%")

    ax.set_ylabel('Precio USD')
    plt.xticks(rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.grid(True, color='gray', linestyle='-', linewidth=0.2)
    return fig



def daily_returns(df):
    df = df.sort_index(ascending=True)
    df['returns'] = np.log(df['close']).diff()
    return df

def returns_vol(df):
    df['volatility'] = df.returns.rolling(12).std()
    return df

def plot_volatility(df_vol):
    background = plt.imread('assets/DFlogo_source.png')
    logo = plt.imread('assets/DFanalytics_logo_plot.png')
    font = {'family': 'sans-serif',
            'color':  'white',
            'weight': 'normal',
            'size': 16,
            }

    font_sub = {'family': 'sans-serif',
            'color':  'white',
            'weight': 'normal',
            'size': 10,
            }


    df_plot = df_vol.copy()
    fig = plt.figure(figsize=(10,6))
    plt.plot(df_plot.index, df_plot.returns, color='dodgerblue', linewidth=0.5)
    plt.plot(df_plot.index, df_plot.volatility, color='darkorange', linewidth=1)
    mplcyberpunk.add_glow_effects()
    plt.ylabel('% Porcentaje')
    plt.xticks(rotation=45,  ha='right')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.3f}'))
    #ax.figure.figimage(logo,  10, 1000, alpha=.99, zorder=1)
    ax.figure.figimage(background, 40, 40, alpha=.15, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.grid(True,color='gray', linestyle='-', linewidth=0.2)
    plt.legend(('Retornos Diarios', 'Volatilidad Móvil'), frameon=False)
    return fig


def plot_prophet(data, n_forecast=365):
    # Reiniciamos el índice para que la fecha no quede como índice
    data_prophet = data.reset_index().copy()

    # Aseguramos que las columnas están en minúsculas
    data_prophet.columns = data_prophet.columns.str.lower()

    # Detectamos la columna de fecha y renombramos
    if 'date' in data_prophet.columns and 'close' in data_prophet.columns:
        data_prophet.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
    else:
        raise KeyError("No se encontró columna 'date' y 'close' en los datos")

    # Entrenamos modelo
    m = Prophet()
    m.fit(data_prophet[['ds', 'y']])

    # Forecast
    future = m.make_future_dataframe(periods=n_forecast)
    forecast = m.predict(future)

    # Generamos la figura manualmente para mejorar el estilo
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast['ds'], forecast['yhat'], color="darkorange", linewidth=2, label="Pronóstico")
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                color="deepskyblue", alpha=0.2, label="Intervalo Confianza")
    ax.plot(data_prophet['ds'], data_prophet['y'], color="dodgerblue", linewidth=1, label="Histórico")


    # Fondo y estilos
    try:
        background = plt.imread('assets/DFlogo_source.png')
        fig.figimage(background, 40, 40, alpha=.15, zorder=1)
    except FileNotFoundError:
        print("⚠️ Logo no encontrado en assets/")

    mplcyberpunk.add_glow_effects()
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(True, color='gray', linestyle='-', linewidth=0.4)
    ax.set_ylabel('Precio de Cierre')
    plt.xticks(rotation=45, ha='right')


    return fig



###########################
#### LAYOUT - Sidebar
###########################

DFanalytics_logo = Image.open('assets/DFanalytics_logo_plot.png')
with st.sidebar:
    st.image(DFanalytics_logo)
    stock = st.selectbox('Ticker', ['GMM','MPWR','MASK','PTGX','SIGA','TRMD','PDD','QSG','FTLF','VGZ'], index=1)
    start_time = st.date_input(
                    "Fecha de Inicio",
                    datetime.date(2022, 7, 6))
    end_time = st.date_input(
                    "Fecha Final",
                    datetime.date(2025, 8, 8))
    periods = st.number_input('Periodos Forecast', value=365, min_value=1, max_value=5000)


###########################
#### DATA - Funciones sobre inputs
###########################

data = get_data(stock, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
plot_price = plot_close_price(data)

df_ret = daily_returns(data)
df_vol = returns_vol(df_ret)
plot_vol = plot_volatility(df_vol)

plot_forecast = plot_prophet(data, periods)



###########################
#### LAYOUT - Render Final
###########################

st.title("Análisis de Acciones")

st.subheader('Precio de Cierre - Fibonacci')
st.pyplot(plot_price)

st.subheader('Forecast a un Año - Prophet')
st.pyplot(plot_forecast)

st.subheader('Retornos Diarios')
st.pyplot(plot_vol)

st.dataframe(data)