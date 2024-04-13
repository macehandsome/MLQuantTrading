import json
import redis
from risk_manager import get_current_balance

def update_targetPNL_stoplossPNL(instrument=None,current_pnl=None):
    r = redis.Redis(host='localhost', port=6379, db=0)
    instruments = r.lrange('instruments', 0, -1)
    instruments = [i.decode('utf-8') for i in instruments]
    risk_factors = json.loads(r.get('risk_factors').decode('utf-8'))
    risk_rewards = json.loads(r.get('risk_rewards').decode('utf-8'))
    # per_instrument_balance = get_current_balance() / len(instruments)
    per_instrument_balance = get_current_balance() 

    for ins in instruments:
        risk_factor = risk_factors[ins]
        risk_reward = risk_rewards[ins]
        stoploss_pnl = per_instrument_balance * risk_factor
        target_pnl = stoploss_pnl * risk_reward

        r.hset(ins, 'target_pnl', target_pnl)
        r.hset(ins, 'stoploss_pnl', stoploss_pnl)
    if instrument is not None :
        r.hset(instrument, 'current_pnl', current_pnl)

def load_TargetPNL_stoplossPNL(instrument):
    r = redis.Redis(host='localhost', port=6379, db=0)
    details = r.hgetall(instrument)
    if details:
        target_pnl = float(details[b'target_pnl'].decode('utf-8'))
        stoploss_pnl = float(details[b'stoploss_pnl'].decode('utf-8'))
        return instrument,target_pnl,stoploss_pnl
    return None

def load_data():
    r = redis.Redis(host='localhost', port=6379, db=0)
    instruments = r.lrange('instruments', 0, -1)
    instruments = [i.decode('utf-8') for i in instruments]
    data = []
    for ins in instruments:
        details = r.hgetall(ins)
        if details:
            data.append({
                'Instrument': ins,
                'Target PNL': float(details[b'target_pnl'].decode('utf-8')),
                'Current PNL': float(details[b'current_pnl'].decode('utf-8')) if b'current_pnl' in details else 0,
                'Stoploss PNL': float(details[b'stoploss_pnl'].decode('utf-8'))
            })
    return data