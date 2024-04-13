from signal_generator import generate_signal
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

    inposition = r.get('inposition').decode('utf-8') == 'True'

    stoploss, takeprofit, quantity = get_quantity(instrument,trade_direction)
   
    print("==" * 25)
    print("Oanda quantities") 
    print("Instrument:", instrument, " | Vol :", quantity, " | StopLoss :", stoploss, " | Takeprofit :", takeprofit)         
    #Place orders
    place_market_order(instrument, quantity, takeprofit, stoploss)
    r.hset(instrument, 'inposition', 'True')
    # inposition = True      
    time.sleep(3)


def close_all_trades_withCurrentAccount():
    close_all_trades(client, accountID)

def start_trading():

    # global opening_balance
    # global target_pnl
    # global stoploss_pnl
    # global inposition
    global instrument
    global lookback_count
    global stma_period
    global ltma_period
    global client
    global accountID
    global last_print_time

    global risk_factors
    global risk_rewards
    print("Trading started")
    # is_trading_running = True
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('is_trading_running', 'True')
    
    print("==" * 25)
    print("")
    print("==" * 25)
    print("Starting balance : {:.2f}".format(get_current_balance()))
    # print("Take Profit initial : {:.2f}".format(target_pnl))
    # print("Stop loss initial : {:.2f}".format(stoploss_pnl))
    
    print("==" * 25)             
    while  True:    
        if (r.get('is_trading_running').decode('utf-8') == 'True'):
            try:
                # we will trade only if NOT in position
                
                inposition = r.hgetall(instrument)[b'inposition'].decode('utf-8') == 'True'
                if inposition == False:
                    trade_direction = generate_signal(instrument, lookback_count, stma_period, ltma_period)
                
                    if trade_direction is None:
                        pass
                    else:
                        print("Found opportunity in {}".format(instrument))
                        find_quantities_and_trade(instrument,trade_direction)
                        #send_email_notification()  
                        

                if inposition == True:    
                    positions_dict = get_open_positions()
                    long_pnl, short_pnl, total_pnl = calculate_total_unrealised_pnl(positions_dict)    
                    _,target_pnl,stoploss_pnl = load_TargetPNL_stoplossPNL('EUR_USD')
                    current_time = time.time()
                    #check pnl
                    update_targetPNL_stoplossPNL(instrument,total_pnl)
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
                        opening_balance = get_current_balance()

                        update_targetPNL_stoplossPNL(instrument,0)
                        subject = "Closing Trades"
                        body = msg
                        #send_email_notification(subject, body)                

                    else:      
                        pass
            except  Exception as e:
                    print("Error in trading",e)
                    pass

            time.sleep(3)
        else:
            time.sleep(3)
    try :
        close_all_trades(client, accountID)
    finally:
        print("Trading Stopped")
        r.set('is_trading_running', 'False')
        print("==" * 25)

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


    #helper variables
    # instruments = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]

    # lookback_count = 200
    # stma_period = 9
    # ltma_period = 20
    # inposition = False
    is_trading_running = r.get('is_trading_running').decode('utf-8')
    lookback_count = int(r.get('lookback_count').decode('utf-8'))
    stma_period = int(r.get('stma_period').decode('utf-8'))
    ltma_period = int(r.get('ltma_period').decode('utf-8'))
    # inposition = r.get('inposition').decode('utf-8') == 'True'

    instruments = r.lrange('instruments', 0, -1)
    instruments = [i.decode('utf-8') for i in instruments]

    risk_factors = json.loads(r.get('risk_factors').decode('utf-8'))
    risk_rewards = json.loads(r.get('risk_rewards').decode('utf-8'))



    update_targetPNL_stoplossPNL()
    # instrument = 'EUR_USD'
    # instrument = 'USD_CHF'
    # instrument = "AUD_USD"

    instrument = 'USD_CHF'


    last_print_time = time.time()
    time_interval = 1*60

    # update_targetPNL_stoplossPNL()
    print("Starting Trading at backend")
    start_trading()
