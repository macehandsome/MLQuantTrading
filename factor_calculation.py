#factor calculation



def std(data):
    data['sd_y'] = data['Close'].rolling(252).std()
    data['sd_m'] = data['Close'].rolling(21).std()
    data['sd_w'] = data['Close'].rolling(5).std()

def bollinger_strat(data, window, no_of_std):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()

    data['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
    data['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)

def heikin_ashi(data):
    data['open_HA'] = (data['Open'].shift(1) + data['Close'].shift(1))/2
    data['close_HA'] = (data['open_HA']+data['Close']+data['High']+data['Low'])/4

def calculate_stochastic_oscillator(data, n=14, m=3):
    high_n = data['High'].rolling(window=n).max()
    low_n = data['Low'].rolling(window=n).min()
    k = ((data['Close'] - low_n) / (high_n - low_n)) * 100
    d = k.rolling(window=m).mean()
    return k, d

def calculate_bears_bulls_power(data, ma_window=14):
    ma = data['Close'].rolling(window=ma_window).mean()
    bulls_power = data['High'] - ma
    bears_power = data['Low'] - ma
    return bears_power, bulls_power

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
