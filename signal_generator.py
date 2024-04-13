import numpy as np
from oandapyV20 import API
import pandas as pd
#API processes requests that can be created fro the endpoints
from oandapyV20.endpoints.instruments import InstrumentsCandles

from oandapyV20.endpoints.pricing import PricingInfo

from Training import fetch_candlestick_data
access_token = "579cc72575476a0d554923af714817c7-762cabdedada32f2be92a785df983c3d"
account_id = "101-003-28602471-001"

accountID = account_id
access_token = access_token


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