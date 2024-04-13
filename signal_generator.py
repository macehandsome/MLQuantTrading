import numpy as np
from oandapyV20 import API
#API processes requests that can be created fro the endpoints
from oandapyV20.endpoints.instruments import InstrumentsCandles

from oandapyV20.endpoints.pricing import PricingInfo


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
        'granularity': 'H1',
        'price': 'M',  # Midpoint candlestick prices
    }

    # Request the candlestick data from Oanda API
    candles_request = InstrumentsCandles(instrument=instrument_name, params=params)
    response = api.request(candles_request)

    # Extract the close prices from the response
    close_prices = [float(candle['mid']['c']) for candle in response['candles']]

    return close_prices

def generate_signal(instrument_name, lookback_count, stma_period, ltma_period):
    # Fetch candlestick data
    close_prices = fetch_candlestick_data(instrument_name, lookback_count)

    # Calculate short-term moving average (STMA)
    stma = np.mean(close_prices[-stma_period:])

    # Calculate long-term moving average (LTMA)
    ltma = np.mean(close_prices[-ltma_period:])
    # Check for crossover
    if stma > ltma:
        signal = "BUY"
    else:
        signal = "SELL"

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