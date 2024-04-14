import oandapyV20
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.exceptions import V20Error
from notification import send_email_notification
import redis


access_token = "579cc72575476a0d554923af714817c7-762cabdedada32f2be92a785df983c3d"
account_id = "101-003-28602471-001"

accountID = account_id
access_token = access_token
client = oandapyV20.API(access_token=access_token, environment="practice")

def get_current_price(instrument):
    params = {
        "instruments": instrument
    }
    request = pricing.PricingInfo(accountID=account_id, params=params)
    response = client.request(request)

    if 'prices' in response and response['prices']:
        return float(response['prices'][0]['bids'][0]['price'])

    return None

def get_instrument_precision(instrument):
    instrument_precision = {
        "EUR_USD":4,
        "AUD_USD":4,
        "NZD_USD":4,
        "GBP_USD":4,
        "GBP_JPY":2,
        "USD_JPY":2,
        "EUR_JPY":2
        }
    ans = instrument_precision.get(instrument) if instrument in instrument_precision else 4
    return ans  # Set a default precision value if instrument not found


def get_current_balance():
    request = AccountDetails(accountID=accountID)
    response = client.request(request)

    if response and 'account' in response:
        account_info = response['account']
        balance = float(account_info['balance'])
        return balance
        
    
    return None

def get_account_equity():
    request = AccountDetails(accountID=accountID)
    response = client.request(request)

    if response and 'account' in response:
        account_info = response['account']
        balance = float(account_info['balance'])
        unrealized_pnl = float(account_info['unrealizedPL'])  
        equity = balance + unrealized_pnl  
        return equity
    
def get_quantity(instrument, trade_direction):
    current_price = get_current_price(instrument)
    take_profit_percentage = 0.02
    stop_loss_percentage = 0.02

    print('get_instrument_precision(instrument)',get_instrument_precision(instrument))
    if trade_direction == "BUY":
        take_profit_price = round(current_price * (1 + take_profit_percentage), get_instrument_precision(instrument))
        stop_loss_price = round(current_price * (1 - stop_loss_percentage), get_instrument_precision(instrument))
    elif trade_direction == "SELL":
        take_profit_price = round(current_price * (1 - take_profit_percentage), get_instrument_precision(instrument))
        stop_loss_price = round(current_price * (1 + stop_loss_percentage), get_instrument_precision(instrument))
    else:
        print("Invalid trade direction")
        return
    #print(stop_loss_price, take_profit_price)

    trade_currency_2 = instrument
    position_size=None
    # A more complex calculation can be done 

    if "AUD" in trade_currency_2:
        position_size = 20000
    elif "JPY" in trade_currency_2:
        position_size = 20000
    elif "CHF" in trade_currency_2:
        position_size = 20000
    elif "EUR" in trade_currency_2:
        position_size = 20000
    elif "GBP" in trade_currency_2:
        position_size = 20000
    else:
        print("Unsupported currency in the denominator")
        position_size = None
    
    if trade_direction == "BUY":
        position_size = position_size 
    else:
        position_size = -position_size

    return stop_loss_price, take_profit_price, position_size


def get_open_positions():
    request = positions.OpenPositions(accountID=account_id)
    response = client.request(request)
    open_positions = response.get("positions", [])
    return open_positions

def calculate_total_unrealised_pnl(positions_dict,instrument_name):
    # long_pnl = 0
    # short_pnl = 0
    # total_pnl = 0

    # for position in positions_dict:
    #     long_unrealized_pnl = float(position['long']['unrealizedPL'])
    #     short_unrealized_pnl = float(position['short']['unrealizedPL'])

    #     long_pnl += long_unrealized_pnl
    #     short_pnl += short_unrealized_pnl
    #     total_pnl = long_pnl + short_pnl
    long_pnl = 0
    short_pnl = 0
    total_pnl = 0

    for position in positions_dict:
        if position['instrument'] == instrument_name:
            long_unrealized_pnl = float(position['long']['unrealizedPL'])
            short_unrealized_pnl = float(position['short']['unrealizedPL'])

            long_pnl += long_unrealized_pnl
            short_pnl += short_unrealized_pnl
            total_pnl = long_pnl + short_pnl
            break

    return long_pnl, short_pnl, total_pnl

def get_current_balance():
    request = AccountDetails(accountID=accountID)
    response = client.request(request)

    if response and 'account' in response:
        account_info = response['account']
        balance = float(account_info['balance'])
        return balance

def place_market_order(instrument, units, take_profit_price, stop_loss_price):
    data = {
        "order": {
            "units": str(units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {
                "price": str(float(take_profit_price)),
            },
            "stopLossOnFill": {
                "price": str(float(stop_loss_price)),
            }
        }
    }
    
    try:
        request = orders.OrderCreate(accountID, data=data)
        response = client.request(request)
        print("Oanda Orders placed successfully!")
        subject = "Oanda Trades Initiated"
        body = "Oanda Trades Initiated"
        #send_email_notification(subject, body)
    except V20Error as e:
        print("Error placing Oanda orders:")
        print(e)
        subject = "Failed to Take Oanda Trades"
        body = "Failed to Take Oanda Trades"
        #send_email_notification(subject, body)



def close_all_trades(client, account_id,instrument = 'ALL'):
    # Get a list of all open trades for the account
    trades_request = trades.OpenTrades(accountID=account_id)
    response = client.request(trades_request)
    r = redis.Redis(host='localhost', port=6379, db=0)

    if len(response['trades']) > 0:
        for trade in response['trades']:
            trade_instrument = trade['instrument']
            trade_id = trade['id']
            # Only close trades that match the specified instrument, if any
            if instrument == 'ALL' or trade_instrument == instrument:
                try:
                    # Create a market order to close the trade
                    data = {"units": "ALL"}
                    order_request = trades.TradeClose(accountID=account_id, tradeID=trade_id, data=data)
                    response = client.request(order_request)
                    print(f"Trade {trade_id} for {trade_instrument} closed successfully.")
                except oandapyV20.exceptions.V20Error as e:
                    print(f"Failed to close trade {trade_id} for {trade_instrument}. Error: {e}")
            else:
                continue
                # print(f"Skipping trade {trade_id} for {trade_instrument}, not matching {instrument}.")
        if instrument == 'ALL':
            for ins in r.lrange('instruments', 0, -1):
                instrument = ins.decode('utf-8')
                r.hset(instrument, 'inposition', 'False')
                r.hset(instrument, 'target_pnl', 0)
                r.hset(instrument, 'stoploss_pnl', 0)
                r.hset(instrument, 'current_pnl', 0)
        else:
            r.hset(instrument, 'inposition', 'False')
            r.hset(instrument, 'target_pnl', 0)
            r.hset(instrument, 'stoploss_pnl', 0)
            r.hset(instrument, 'current_pnl', 0)

    else:
        print("No open trades to close.")