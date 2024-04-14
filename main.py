from signal_generator import generate_signal_1
from signal_generator import generate_signal_2
from risk_manager import get_quantity,  place_market_order, get_open_positions, get_current_balance, calculate_total_unrealised_pnl, close_all_trades
from notification import send_email_notification

from utli import update_targetPNL_stoplossPNL,load_TargetPNL_stoplossPNL
import time
import oandapyV20

import redis
import json
from config import init_paramters




def find_quantities_and_trade(instrument, trade_direction):
    # global takeprofit
    # global stoploss
    # global inposition

    stoploss, takeprofit, quantity = get_quantity(instrument,trade_direction)

    print("==" * 25)
    print("Oanda quantities")
    print("Instrument:", instrument, " | Vol :", quantity, " | StopLoss :", stoploss, " | Takeprofit :", takeprofit)
    #Place orders
    place_market_order(instrument, quantity, takeprofit, stoploss)
    r.hset(instrument, 'inposition', 'True')
    # inposition[instrument] = True
    time.sleep(3)


def close_all_trades_withCurrentAccount():
    close_all_trades(client, accountID)

def start_trade(instrument,lookback_count,selected_strat):
    global inposition
    access_token = "579cc72575476a0d554923af714817c7-762cabdedada32f2be92a785df983c3d"
    account_id = "101-003-28602471-001"

    accountID = account_id
    access_token = access_token
    client = oandapyV20.API(access_token=access_token, environment="practice")

    opening_balance = get_current_balance()
    r.set('is_trading_running', 'True')
    # risk_factor = 0.016 / 100
    # stoploss_pnl = opening_balance * risk_factor
    # risk_reward = 0.75  # 3/4
    # target_pnl =  stoploss_pnl * risk_reward


    last_print_time = time.time()
    time_interval = 1*608008
    while True:
        if (r.get('is_trading_running').decode('utf-8') == 'True'):
            try:
                # we will trade only if NOT in position
                inposition = r.hgetall(instrument)[b'inposition'].decode('utf-8') == 'True'
                selected_strat = int(r.get('strategy').decode('utf-8'))
                if inposition == False:
                    if selected_strat == 1:
                        trade_direction = generate_signal_1(instrument, lookback_count)
                    else:
                        trade_direction = generate_signal_2(instrument, lookback_count)

                    if trade_direction is None:
                        pass
                    else:
                        print("Found opportunity in {}".format(instrument))
                        find_quantities_and_trade(instrument,trade_direction)
                        #send_email_notification()


                if inposition == True:
                    positions_dict = get_open_positions()
                    long_pnl, short_pnl, total_pnl = calculate_total_unrealised_pnl(positions_dict, instrument)
                    _,target_pnl,stoploss_pnl = load_TargetPNL_stoplossPNL(instrument)

                    current_time = time.time()
                    update_targetPNL_stoplossPNL(instrument,total_pnl)
                    #check pnl
                    if current_time - last_print_time >= time_interval:
                        print(f" Target:  {target_pnl:.2f} | StopLoss: {stoploss_pnl :.2f} | PNL:  {total_pnl:.2f} ")
                        last_print_time = current_time
                    #exit check
                    if (total_pnl > target_pnl) or total_pnl < -(stoploss_pnl):
                        if (total_pnl > target_pnl):
                            msg = f"Profit Trade, Target : {target_pnl:.2f} | Actual: {total_pnl:.2f}"
                        elif total_pnl < -(stoploss_pnl):
                            msg = f"Loss Trade, Target:  {target_pnl:.2f} | Actual: {total_pnl:.2f} "
                        print(msg)
                        close_all_trades(client, accountID,instrument)
                        print("Closing all Trades")
                        print("Current balance: {:.2f}".format(get_current_balance()))
                        r.hset(instrument, 'inposition', 'False')
                        update_targetPNL_stoplossPNL(instrument,0)
                        # inposition[instrument] = False
                        # opening_balance = get_current_balance()
                        # risk_factor = 0.016 / 100
                        # stoploss_pnl = opening_balance * risk_factor
                        # risk_reward = 0.75
                        # target_pnl =  stoploss_pnl * risk_reward

                        subject = "Closing Trades"
                        body = msg


                    #send_email_notification(subject, body)


                    else:
                        pass
            except:
                    pass
            time.sleep(2)
        else:
            time.sleep(3)

    time.sleep(5)


def login():
    access_token = "579cc72575476a0d554923af714817c7-762cabdedada32f2be92a785df983c3d"
    account_id = "101-003-28602471-001"

    accountID = account_id
    access_token = access_token
    client = oandapyV20.API(access_token=access_token, environment="practice")
    return accountID, client
if __name__ == "__main__":
    #use redis to store variables
    global accountID
    global client
    global r
    accountID, client = login()
    r = redis.Redis(host='localhost', port=6379, db=0)
    init_paramters()
    update_targetPNL_stoplossPNL()
    import threading
    from functools import partial
    is_trading_running = r.get('is_trading_running').decode('utf-8')

    instruments = r.lrange('instruments', 0, -1)
    instruments = [i.decode('utf-8') for i in instruments]

    risk_factors = json.loads(r.get('risk_factors').decode('utf-8'))
    risk_rewards = json.loads(r.get('risk_rewards').decode('utf-8'))

    update_targetPNL_stoplossPNL()
    # instruments = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]
    # inposition = {}
    # inposition["USD_CHF"] = False
    # inposition["AUD_USD"] = False
    # inposition["EUR_USD"] = False
    # inposition["GBP_USD"] = False
    # inposition["USD_JPY"] = False
    currency_pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]

    threads = []
    for currency in currency_pairs:
        thread = threading.Thread(target=partial(start_trade, currency, 252, 1))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


