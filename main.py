import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Stock Prediction App')

stocks_dict = {
    'Rio Tinto':'RIO',
    'BHP':'BHP',
    'Newmont Goldcorp':'NEM',
    'Freeport McMoran':'FCX',
    'Glencore':'GLEN.L',
    'Hudbay Minerals':'HBM',
    'Nexa Resources':'NEXA',
    'MMG Limited':'1208.HK'
    }

stocks_keys = list(stocks_dict.keys())

selected_stock_key = st.sidebar.selectbox('Select dataset for prediction', stocks_keys)

selected_stock = stocks_dict[selected_stock_key]

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# data_load_state = st.text('Load data')
data = load_data(selected_stock)
# data_load_state = st.text('Loading data...')

st.subheader("Raw data")

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock close'))
    fig.layout.update(title_text=f'{selected_stock_key}: {selected_stock}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

st.write(data.iloc[::-1].head())

# Predict forecast
df_train = data[['Date', 'Close']]
df_train.columns = ['ds', 'y']

# Show and plot forecast
st.subheader('Forecast data')
n_years = st.slider('Years of prediction: ', 1, 4)
period = n_years * 365

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods = period)
forecast = model.predict(future)

st.write(forecast.iloc[::-1].head())

fig1 = plot_plotly(model, forecast)
fig1.layout.update(title_text=f'Forecast plot for {n_years} years', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.subheader('Forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)