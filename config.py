# # config.py
# status = {
#     'is_trading_running': False,
#     'instruments': ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"],
#     'lookback_count': 200,
#     'stma_period': 9,
#     'ltma_period': 20,
#     'inposition': False,
#     'risk_factors': {
#         "EUR_USD": 0.016 / 100,
#         "USD_JPY": 0.018 / 100,
#         "GBP_USD": 0.015 / 100,
#         "USD_CHF": 0.017 / 100,
#         "AUD_USD": 0.019 / 100,
#     },
#     'risk_rewards': {
#         "EUR_USD": 0.75,
#         "USD_JPY": 0.8,
#         "GBP_USD": 0.7,
#         "USD_CHF": 0.65,
#         "AUD_USD": 0.85,
#     }
# }
import redis
import json  

def init_paramters():
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.flushdb()

    r.set('is_trading_running', 'False')
    r.set('lookback_count', '200')
    # r.set('stma_period', '9')
    # r.set('ltma_period', '20')
    r.set('inposition', 'False')
    r.set('strategy','1')


    instruments = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]
    r.lpush('instruments', *instruments)
    for ins in instruments:
        r.hset(ins, 'inposition', 'False')
        # r.hset(ins, 'target_pnl', 0)
        # r.hset(ins, 'stoploss_pnl', 0)
        # r.hset(ins, 'current_pnl', 0)
    risk_factors = {
        "EUR_USD": 0.016 / 100,
        "USD_JPY": 0.018 / 100,
        "GBP_USD": 0.015 / 100,
        "USD_CHF": 0.017 / 100,
        "AUD_USD": 0.019 / 100,
    }

    risk_rewards = {
        "EUR_USD": 0.75,
        "USD_JPY": 0.8,
        "GBP_USD": 0.7,
        "USD_CHF": 0.65,
        "AUD_USD": 0.85,
    }

    r.set('risk_factors', json.dumps(risk_factors))
    r.set('risk_rewards', json.dumps(risk_rewards))