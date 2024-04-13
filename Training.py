import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.inspection import permutation_importance
scaler = preprocessing.MinMaxScaler()

def process_csv(file_path):
    # open
    data = pd.read_csv(file_path)

    # calculate log returns. Next day's returns will be used for Y
    data['return'] = np.log(1+ data['Close'].pct_change())
    data['Y'] = data['return'].shift(-1)
    data.dropna(inplace=True)

    # Scale the DataFrame
    data_scaled = scaler.fit_transform(data.drop(['Date','Y'],axis=1))
    data_scaled = pd.DataFrame(data_scaled, columns=['Open','High','Low','Close','Adj Close','Volume','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return'])

    # Concatenate the scaled DataFrame with the 'Date' column
    data = pd.concat([data[['Date','Y']], data_scaled], axis=1)

    return data

eur = process_csv('usdeur.csv')
jpy = process_csv('usdjpy.csv')
gbp = process_csv('usdgbp.csv')
aud = process_csv('usdaud.csv')
chf = process_csv('usdchf.csv')

def train_test_split_by_date(data, split_date, name):
    data['Date'] = pd.to_datetime(data['Date'])

    # Split the DataFrame into training and testing sets
    x = data[data['Date'] <= split_date].sort_values('Date')
    y = data[data['Date'] <= split_date].sort_values('Date')['Y']
    test = data[data['Date'] > split_date].sort_values('Date')

    # Store the sets in a dictionary with their names
    sets = {
        f'{name}train_x': x,
        f'{name}train_y': y,
        f'{name}test': test
    }

    return sets

split_date = '2021-12-31'
eur_split = train_test_split_by_date(eur, split_date, '')
jpy_split = train_test_split_by_date(jpy, split_date, '')
gbp_split = train_test_split_by_date(gbp, split_date, '')
aud_split = train_test_split_by_date(aud, split_date, '')
chf_split = train_test_split_by_date(chf, split_date, '')

from sklearn.model_selection import TimeSeriesSplit

n_fold = 5

folds = TimeSeriesSplit(
    n_splits=n_fold,
    gap=1, # We use a gap of 1 day between the train and test side of the splits.
    max_train_size=10000,
    test_size=287, # we sample 287 rows of data, which is about 10% of training data
)

params = {#'num_leaves': 555,
          #'min_child_weight': 0.034,
          #'feature_fraction': 0.379,
          #'bagging_fraction': 0.418,
          #'min_data_in_leaf': 106,
          #'objective': 'regression',
          #'max_depth': -1,
          #'learning_rate': 0.005,
          "boosting_type": "gbdt",
          #"bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          #'reg_alpha': 0.3899,
          #'reg_lambda': 0.648,
          'random_state': 13,
         }
from sklearn import preprocessing, metrics

# eur first
columns = ['High','Low','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return']
splits = folds.split(eur_split['train_x'], eur_split['train_y'])
y_preds = np.zeros(eur_split['test'].shape[0])
y_oof = np.zeros(eur_split['train_x'].shape[0])
feature_importances_eur = pd.DataFrame()
feature_importances_eur['feature'] = columns
mean_score = []
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = eur_split['train_x'][columns].iloc[train_index], eur_split['train_x'][columns].iloc[valid_index]
    y_train, y_valid = eur_split['train_y'].iloc[train_index], eur_split['train_y'].iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    clf_eur = lgb.train(params, dtrain, 2500, valid_sets = [dtrain, dvalid])

    feature_importances_eur[f'fold_{fold_n + 1}'] = clf_eur.feature_importance()
    y_pred_valid = clf_eur.predict(X_valid,num_iteration=clf_eur.best_iteration)
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    y_preds += clf_eur.predict(eur_split['test'][columns], num_iteration=clf_eur.best_iteration)/n_fold
print('mean rmse score over folds is',np.mean(mean_score))
eur_split['test']['predicted'] = y_preds
eur_predicted = eur_split['test'][['Date','return','Y','predicted']]

# jpy
columns = ['High','Low','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return']
splits = folds.split(jpy_split['train_x'], jpy_split['train_y'])
y_preds = np.zeros(jpy_split['test'].shape[0])
y_oof = np.zeros(jpy_split['train_x'].shape[0])
feature_importances_jpy = pd.DataFrame()
feature_importances_jpy['feature'] = columns
mean_score = []
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = jpy_split['train_x'][columns].iloc[train_index], jpy_split['train_x'][columns].iloc[valid_index]
    y_train, y_valid = jpy_split['train_y'].iloc[train_index], jpy_split['train_y'].iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    clf_jpy = lgb.train(params, dtrain, 2500, valid_sets = [dtrain, dvalid])

    feature_importances_jpy[f'fold_{fold_n + 1}'] = clf_jpy.feature_importance()
    y_pred_valid = clf_jpy.predict(X_valid,num_iteration=clf_jpy.best_iteration)
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    y_preds += clf_jpy.predict(jpy_split['test'][columns], num_iteration=clf_jpy.best_iteration)/n_fold
print('mean rmse score over folds is',np.mean(mean_score))
jpy_split['test']['predicted'] = y_preds
jpy_predicted = jpy_split['test'][['Date','return','Y','predicted']]

# gbp
columns = ['High','Low','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return']
splits = folds.split(gbp_split['train_x'], gbp_split['train_y'])
y_preds = np.zeros(gbp_split['test'].shape[0])
y_oof = np.zeros(gbp_split['train_x'].shape[0])
feature_importances_gbp = pd.DataFrame()
feature_importances_gbp['feature'] = columns
mean_score = []
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = gbp_split['train_x'][columns].iloc[train_index], gbp_split['train_x'][columns].iloc[valid_index]
    y_train, y_valid = gbp_split['train_y'].iloc[train_index], gbp_split['train_y'].iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    clf_gbp = lgb.train(params, dtrain, 2500, valid_sets = [dtrain, dvalid])

    feature_importances_gbp[f'fold_{fold_n + 1}'] = clf_gbp.feature_importance()
    y_pred_valid = clf_gbp.predict(X_valid,num_iteration=clf_gbp.best_iteration)
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    y_preds += clf_gbp.predict(gbp_split['test'][columns], num_iteration=clf_gbp.best_iteration)/n_fold
print('mean rmse score over folds is',np.mean(mean_score))
gbp_split['test']['predicted'] = y_preds
gbp_predicted = gbp_split['test'][['Date','return','Y','predicted']]

# aud
columns = ['High','Low','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return']
splits = folds.split(aud_split['train_x'], aud_split['train_y'])
y_preds = np.zeros(aud_split['test'].shape[0])
y_oof = np.zeros(aud_split['train_x'].shape[0])
feature_importances_aud = pd.DataFrame()
feature_importances_aud['feature'] = columns
mean_score = []
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = aud_split['train_x'][columns].iloc[train_index], aud_split['train_x'][columns].iloc[valid_index]
    y_train, y_valid = aud_split['train_y'].iloc[train_index], aud_split['train_y'].iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    clf_aud = lgb.train(params, dtrain, 2500, valid_sets = [dtrain, dvalid])

    feature_importances_aud[f'fold_{fold_n + 1}'] = clf_aud.feature_importance()
    y_pred_valid = clf_aud.predict(X_valid,num_iteration=clf_aud.best_iteration)
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    y_preds += clf_aud.predict(aud_split['test'][columns], num_iteration=clf_aud.best_iteration)/n_fold
print('mean rmse score over folds is',np.mean(mean_score))
aud_split['test']['predicted'] = y_preds
aud_predicted = aud_split['test'][['Date','return','Y','predicted']]

# chf
columns = ['High','Low','sd_y','sd_m','sd_w','Bollinger High','Bollinger Low','open_HA','close_HA','Stochastic oscillator','Bears Power','Bulls Power','RSI','return']
splits = folds.split(chf_split['train_x'], chf_split['train_y'])
y_preds = np.zeros(chf_split['test'].shape[0])
y_oof = np.zeros(chf_split['train_x'].shape[0])
feature_importances_chf = pd.DataFrame()
feature_importances_chf['feature'] = columns
mean_score = []
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = chf_split['train_x'][columns].iloc[train_index], chf_split['train_x'][columns].iloc[valid_index]
    y_train, y_valid = chf_split['train_y'].iloc[train_index], chf_split['train_y'].iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    clf_chf = lgb.train(params, dtrain, 2500, valid_sets = [dtrain, dvalid])

    feature_importances_chf[f'fold_{fold_n + 1}'] = clf_chf.feature_importance()
    y_pred_valid = clf_chf.predict(X_valid,num_iteration=clf_chf.best_iteration)
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    y_preds += clf_chf.predict(chf_split['test'][columns], num_iteration=clf_chf.best_iteration)/n_fold
print('mean rmse score over folds is',np.mean(mean_score))
chf_split['test']['predicted'] = y_preds
chf_predicted = chf_split['test'][['Date','return','Y','predicted']]

eur_predicted[["Y", "predicted"]].cumsum().plot(figsize=(10,3))
jpy_predicted[["Y", "predicted"]].cumsum().plot(figsize=(10,3))
gbp_predicted[["Y", "predicted"]].cumsum().plot(figsize=(10,3))
aud_predicted[["Y", "predicted"]].cumsum().plot(figsize=(10,3))
chf_predicted[["Y", "predicted"]].cumsum().plot(figsize=(10,3))

import warnings
warnings.filterwarnings("ignore")

def visualizing_strategies(data):
    # We need to be long(short) when Y is positive(negative). Try binary first
    data['difference'] = data['predicted'] - data['Y']
    data['sign_y'] = np.sign(data['Y'])
    data['sign_pred'] = np.sign(data['predicted'])
    data['strat'] = data['sign_pred']*data['Y']

    # use standard deviation as confidence band for trading. Our prediction needs to be higher/lower than this band in order for trade
    # Set the threshold values. Here it's STD/2
    positive_threshold = data['difference'].std()/2
    negative_threshold = -data['difference'].std()/2
    data['sign_pred2'] = np.where((data['predicted'] > positive_threshold), 1,
                                np.where((data['predicted'] < negative_threshold), -1, np.nan))
    data = data.fillna(method='ffill').fillna(method='bfill') # if the 1st row is NaN
    data['strat2'] = data['sign_pred2']*data['Y']

    # Calculate the returns
    benchmark_returns = data['Y'].sum()
    strategy_returns = data['strat'].sum()
    strategy2_returns = data['strat2'].sum()
    print(f"Benchmark returns: {benchmark_returns:.3f}\nStrategy returns: {strategy_returns:.3f}\nStrategy2 returns: {strategy2_returns:.3f}")

    # Plot the cumulative returns
    data[["Y", "strat",'strat2']].cumsum().plot(figsize=(10, 3))
    plt.show()

def stdev(data):
    data['difference'] = data['predicted'] - data['Y']
    positive_threshold = data['difference'].std()
    return positive_threshold

eur_std = stdev(eur_predicted)
jpy_std = stdev(jpy_predicted)
gbp_std = stdev(gbp_predicted)
aud_std = stdev(aud_predicted)
chf_std = stdev(chf_predicted)

visualizing_strategies(eur_predicted)
visualizing_strategies(jpy_predicted)
visualizing_strategies(gbp_predicted)
visualizing_strategies(aud_predicted)
visualizing_strategies(chf_predicted)

import seaborn as sns

def features_importance(feature_importances, folds):
    # Calculate the total feature importance for HA and concat with the original df
    tt = feature_importances.loc[0] + feature_importances.loc[1] + feature_importances.loc[7] + feature_importances.loc[8]
    tt.feature = 'HA_combined'
    tt = pd.DataFrame(tt).T
    feature_importances = pd.concat([feature_importances, tt], ignore_index=True)

    # Calculate the average feature importance over all folds
    feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

    # Plot the features by average importance
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature')
    plt.show()

features_importance(feature_importances_eur, folds)
features_importance(feature_importances_jpy, folds)
features_importance(feature_importances_gbp, folds)
features_importance(feature_importances_aud, folds)
features_importance(feature_importances_chf, folds)


from factor_calculation import std
from factor_calculation import bollinger_strat
from factor_calculation import heikin_ashi
from factor_calculation import calculate_stochastic_oscillator
from factor_calculation import calculate_bears_bulls_power
from factor_calculation import calculate_rsi
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

dict={}
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
    y_preds = dict[instrument_name][0].predict(data_fx,num_iteration=clf_jpy.best_iteration)
    data_fx['predicted'] = y_preds
    data_fx['signal'] = np.sign(data_fx['predicted'])

    # use standard deviation as confidence band for trading. Our prediction needs to be higher/lower than this band in order for trade
    # Set the threshold values. Here it's STD/2
    positive_threshold = dict[instrument_name][1]/2
    negative_threshold = -dict[instrument_name][1]/2
    data_fx['signal2'] = np.where((data_fx['predicted'] > positive_threshold), 1,
                                np.where((data_fx['predicted'] < negative_threshold), -1, 0))
    return data_fx