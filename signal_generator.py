import numpy as np
from oandapyV20 import API
import pandas as pd
#API processes requests that can be created fro the endpoints
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pickle
from oandapyV20 import API
from factor_calculation import std
from factor_calculation import calculate_rsi
from factor_calculation import calculate_stochastic_oscillator
from factor_calculation import calculate_bears_bulls_power
from factor_calculation import heikin_ashi
from factor_calculation import bollinger_strat
from oandapyV20.endpoints.pricing import PricingInfo


access_token = "579cc72575476a0d554923af714817c7-762cabdedada32f2be92a785df983c3d"
account_id = "101-003-28602471-001"

accountID = account_id
access_token = access_token





dict={}
with open('clf_chf_model.pkl', 'rb') as f:
    clf_chf = pickle.load(f)
with open('clf_eur_model.pkl', 'rb') as f:
    clf_eur = pickle.load(f)
with open('clf_gbp_model.pkl', 'rb') as f:
    clf_gbp = pickle.load(f)
with open('clf_jpy_model.pkl', 'rb') as f:
    clf_jpy = pickle.load(f)
with open('clf_aud_model.pkl', 'rb') as f:
    clf_aud = pickle.load(f)

with open('gbp_std.pkl', 'rb') as f:
    gbp_std = pickle.load(f)
with open('jpy_std.pkl', 'rb') as f:
    jpy_std = pickle.load(f)
with open('eur_std.pkl', 'rb') as f:
    eur_std = pickle.load(f)
with open('aud_std.pkl', 'rb') as f:
    aud_std = pickle.load(f)
with open('chf_std.pkl', 'rb') as f:
    chf_std = pickle.load(f)


dict["USD_CHF"]=[clf_chf,chf_std]
dict["USD_JPY"]=[clf_jpy,jpy_std]
dict["AUD_USD"]=[clf_aud,aud_std]
dict["GBP_USD"]=[clf_gbp,gbp_std]
dict["EUR_USD"]=[clf_eur,eur_std]
access_token = "579cc72575476a0d554923af714817c7-762cabdedada32f2be92a785df983c3d"
account_id = "101-003-28602471-001"

accountID = account_id
access_token = access_token
def fetch_candlestick_data(instrument_name, lookback_count):
    # Initialize the Oanda API client

    api = API(access_token=access_token, environment="practice")

    # Define the parameters for the candlestick data request
    params = {
        'count': lookback_count,
        'granularity': 'D',#timetype
        'price': 'M',  # Midpoint candlestick prices
    }


    # Request the candlestick data from Oanda API
    candles_request = InstrumentsCandles(instrument=instrument_name, params=params)
    response = api.request(candles_request)
    data_fx=pd.DataFrame()
    data_fx['Open'] = [float(candle['mid']['o']) for candle in response['candles']]
    data_fx['High'] = [float(candle['mid']['h']) for candle in response['candles']]
    data_fx['Low'] = [float(candle['mid']['l']) for candle in response['candles']]
    data_fx['Close'] = [float(candle['mid']['l']) for candle in response['candles']]
    data_fx['return'] = np.log(1+ data_fx['Close'].pct_change())
    #data_fx['Y'] = data_fx['return'].shift(-1)
    std(data_fx)

    bollinger_strat(data_fx,20,2)

    heikin_ashi(data_fx)

    k, d = calculate_stochastic_oscillator(data_fx)#USD EUR calculation
    data_fx['%K'] = k
    data_fx['%D'] = d
    data_fx['Stochastic oscillator'] = data_fx['%K'] - data_fx['%D']#Stochastic oscillator

    bears_power, bulls_power = calculate_bears_bulls_power(data_fx)
    data_fx['Bears Power'] = bears_power
    data_fx['Bulls Power'] = bulls_power#Bear Bull power

    close_prices = data_fx['Close']
    rsi = calculate_rsi(close_prices)
    data_fx['RSI'] = rsi

    data_fx.drop(columns=['%K', '%D'], inplace=True)#RSI

    #data_fx.dropna(inplace=True)
    columns = ['High','Low','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return']
    data_fx = data_fx[columns]
    data_fx.dropna(inplace=True)
    y_preds = dict[instrument_name][0].predict(data_fx,num_iteration=dict[instrument_name][0].best_iteration)
    data_fx['predicted'] = y_preds
    data_fx['signal'] = np.sign(data_fx['predicted'])

    # use standard deviation as confidence band for trading. Our prediction needs to be higher/lower than this band in order for trade
    # Set the threshold values. Here it's STD/2
    positive_threshold = dict[instrument_name][1]/2
    negative_threshold = -dict[instrument_name][1]/2
    data_fx['signal2'] = np.where((data_fx['predicted'] > positive_threshold), 1,
                                np.where((data_fx['predicted'] < negative_threshold), -1, 0))
    return data_fx


def generate_signal_1(instrument_name, lookback_count):
    data = fetch_candlestick_data(instrument_name, lookback_count)
    last_row =data.iloc[-1]

    if last_row['signal'] == 1:
        signal = "BUY"
    else:
        signal = "SELL"

    return signal

def generate_signal_2(instrument_name, lookback_count):
    data = fetch_candlestick_data(instrument_name, lookback_count)
    last_row =data.iloc[-1]

    if last_row['signal2'] == 1:
        signal = "BUY"
    elif last_row['signal2'] == -1:
        signal = "SELL"
    else:
        signal = None

    return signal



def fetch_data_to_show(instrument_name, lookback_count=200):

    api = API(access_token=access_token, environment="practice")
    params = {
        'count': lookback_count,
        'granularity': 'S5',
        'price': 'M'
    }
    candles_request = InstrumentsCandles(instrument=instrument_name, params=params)
    response = api.request(candles_request)
    return [{
        'x': [candle['time'] for candle in response['candles']],
        'open': [float(candle['mid']['o']) for candle in response['candles']],
        'high': [float(candle['mid']['h']) for candle in response['candles']],
        'low': [float(candle['mid']['l']) for candle in response['candles']],
        'close': [float(candle['mid']['c']) for candle in response['candles']],
        'type': 'candlestick',
    }]

def fetch_latest_prices(instruments):
    api = API(access_token=access_token, environment="practice")
    params = {
        'instruments': ','.join(instruments)
    }
    pricing_request = PricingInfo(accountID=account_id, params=params)
    response = api.request(pricing_request)
    
    latest_prices = []
    for price in response['prices']:
        latest_prices.append({
            'instrument': price['instrument'],
            'time': price['time'],
            'bid': float(price['bids'][0]['price']),
            'ask': float(price['asks'][0]['price']),
            'mid': (float(price['bids'][0]['price']) + float(price['asks'][0]['price'])) / 2
        })
    return latest_prices