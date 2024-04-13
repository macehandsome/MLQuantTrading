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
    global takeprofit
    global stoploss
    global inposition

    stoploss, takeprofit, quantity = get_quantity(instrument,trade_direction)

    print("==" * 25)
    print("Oanda quantities")
    print("Instrument:", instrument, " | Vol :", quantity, " | StopLoss :", stoploss, " | Takeprofit :", takeprofit)
    #Place orders
    place_market_order(instrument, quantity, takeprofit, stoploss)
    inposition[instrument] = True
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
    risk_factor = 0.016 / 100
    stoploss_pnl = opening_balance * risk_factor
    risk_reward = 0.75  # 3/4
    target_pnl =  stoploss_pnl * risk_reward


    last_print_time = time.time()
    time_interval = 1*608008
    while True:
        try:
            # we will trade only if NOT in position
            if inposition[instrument] == False:
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


            if inposition[instrument] == True:
                positions_dict = get_open_positions()
                long_pnl, short_pnl, total_pnl = calculate_total_unrealised_pnl(positions_dict)

                current_time = time.time()
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
                    close_all_trades(client, accountID)
                    print("Closing all Trades")
                    print("Current balance: {:.2f}".format(get_current_balance()))

                    inposition[instrument] = False
                    opening_balance = get_current_balance()
                    risk_factor = 0.016 / 100
                    stoploss_pnl = opening_balance * risk_factor
                    risk_reward = 0.75
                    target_pnl =  stoploss_pnl * risk_reward

                    subject = "Closing Trades"
                    body = msg


                #send_email_notification(subject, body)


                else:
                    pass
        except:
                pass

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

    inposition = {}
    inposition["USD_CHF"] = False
    inposition["AUD_USD"] = False
    inposition["EUR_USD"] = False
    inposition["GBP_USD"] = False
    inposition["USD_JPY"] = False
    thread1 = threading.Thread(target=partial(start_trade, "USD_CHF", 252, 1))
    thread2 = threading.Thread(target=partial(start_trade, "USD_JPY", 252, 1))
    thread3 = threading.Thread(target=partial(start_trade, "AUD_USD", 252, 1))
    thread4 = threading.Thread(target=partial(start_trade, "GBP_USD", 252, 1))
    thread5 = threading.Thread(target=partial(start_trade, "EUR_USD", 252, 1))

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()

    #helper variables
    # instruments = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]

    # lookback_count = 200
    # stma_period = 9
    # ltma_period = 20
    # inposition = False
    #is_trading_running = r.get('is_trading_running').decode('utf-8')
    #lookback_count = int(r.get('lookback_count').decode('utf-8'))
    #stma_period = int(r.get('stma_period').decode('utf-8'))
    #ltma_period = int(r.get('ltma_period').decode('utf-8'))
    # inposition = r.get('inposition').decode('utf-8') == 'True'

    #instruments = r.lrange('instruments', 0, -1)
    #instruments = [i.decode('utf-8') for i in instruments]

    #risk_factors = json.loads(r.get('risk_factors').decode('utf-8'))
    #risk_rewards = json.loads(r.get('risk_rewards').decode('utf-8'))



    #update_targetPNL_stoplossPNL()
    # instrument = 'EUR_USD'
    # instrument = 'USD_CHF'
    # instrument = "AUD_USD"

    #instrument = 'USD_CHF'


    #last_print_time = time.time()
    #time_interval = 1*60

    # update_targetPNL_stoplossPNL()
    #print("Starting Trading at backend")
    #start_trading()
