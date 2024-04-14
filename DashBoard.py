import dash
from dash import dcc, html, Input, Output,dash_table, State
import dash_bootstrap_components as dbc
from FetchData import get_current_balance,get_account_equity, get_open_positions, calculate_total_unrealised_pnl,close_all_trades,get_current_price
# from main import client, accountID
from main import login
from signal_generator import fetch_data_to_show, fetch_latest_prices
from utli import load_data
import plotly.graph_objs as go
import json
import redis
import threading
import datetime
from dash.dependencies import Input, Output
from dash import callback_context
from dateutil import parser


initial_risk_factors = {
    "EUR_USD": 0.016 / 100,
    "USD_JPY": 0.018 / 100,
    "GBP_USD": 0.015 / 100,
    "USD_CHF": 0.017 / 100,
    "AUD_USD": 0.019 / 100,
}
initial_risk_rewards = {
    "EUR_USD": 0.75,
    "USD_JPY": 0.8,
    "GBP_USD": 0.7,
    "USD_CHF": 0.65,
    "AUD_USD": 0.85,
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # 1000ms update
    dbc.Row(dbc.Col(html.H1("Trading Dashboard", id='time-display'), className="mb-4 text-center", width=12,style={'padding-top': '50px'})),

    dbc.Row(dbc.Col(dash_table.DataTable(
        id='price-table',
        columns=[
            {'name': 'Time', 'id': 'time'},
            {'name': 'Instrument', 'id': 'instrument'},
            {'name': 'Bid', 'id': 'bid'},
            {'name': 'Ask', 'id': 'ask'},
            {'name': 'Mid', 'id': 'mid'}
        ],
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': '#007bff',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
        ]
    ), width=12)),

    dbc.Row(dbc.Col(dcc.Dropdown(
        id='instrument-selector',
        options=[{'label': i, 'value': i} for i in ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]],
        value='EUR_USD',  # default value
        style={'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto', 'marginTop': 20}
    ), width=6)),

    dbc.Row([
        dbc.Col(dcc.Graph(id='live-candlestick'), width=12),
        dcc.Interval(
            id='graph-interval',
            interval=20*1000,  # refresh every 20 seconds for graph
            n_intervals=0
        )
    ]),
    dbc.Row(dbc.Col(html.H1("Update Risk Preferences"), className="text-center mb-4")),

    dbc.Row([
        dbc.Col(dbc.Button('Submit Risk Peference', id='submit-button', n_clicks=0, color="primary", className="mt-3"), width=4,style={'margin-left': '20%'}),
        dbc.Col(html.H2(id='output-state', className="mt-3")),
    ], justify="center"),

    dbc.Row([
        dbc.Col([
            dbc.CardGroup([
                dbc.Label('EUR/USD Risk Factor (%)'),
                dbc.Input(id='EUR_USD_risk_factor', type='text', value=initial_risk_factors["EUR_USD"] * 100),
                dbc.Label('EUR/USD Risk Reward'),
                dbc.Input(id='EUR_USD_risk_reward', type='text', value=initial_risk_rewards["EUR_USD"]),
            ]),
            dbc.CardGroup([
                dbc.Label('USD/JPY Risk Factor (%)'),
                dbc.Input(id='USD_JPY_risk_factor', type='text', value=initial_risk_factors["USD_JPY"] * 100),
                dbc.Label('USD/JPY Risk Reward'),
                dbc.Input(id='USD_JPY_risk_reward', type='text', value=initial_risk_rewards["USD_JPY"]),
            ]),
        ], width=6),

        dbc.Col([
            dbc.CardGroup([
                dbc.Label('GBP/USD Risk Factor (%)'),
                dbc.Input(id='GBP_USD_risk_factor', type='text', value=initial_risk_factors["GBP_USD"] * 100),
                dbc.Label('GBP/USD Risk Reward'),
                dbc.Input(id='GBP_USD_risk_reward', type='text', value=initial_risk_rewards["GBP_USD"]),
            ]),
            dbc.CardGroup([
                dbc.Label('USD/CHF Risk Factor (%)'),
                dbc.Input(id='USD_CHF_risk_factor', type='text', value=initial_risk_factors["USD_CHF"] * 100),
                dbc.Label('USD/CHF Risk Reward'),
                dbc.Input(id='USD_CHF_risk_reward', type='text', value=initial_risk_rewards["USD_CHF"]),
            ]),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.CardGroup([
                dbc.Label('AUD/USD Risk Factor (%)'),
                dbc.Input(id='AUD_USD_risk_factor', type='text', value=initial_risk_factors["AUD_USD"] * 100),
                dbc.Label('AUD/USD Risk Reward'),
                dbc.Input(id='AUD_USD_risk_reward', type='text', value=initial_risk_rewards["AUD_USD"]),
            ]),
        ], width=6),
    ]),


    # dbc.Row(dbc.Col(html.Div(id='output-state'), className="mt-4")),
    html.Hr(),
    dbc.Row(dbc.Col(html.H1("Signal Input:"), className="text-center mb-4")),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Standard Deviation"),
                dbc.CardBody([
                    dcc.Markdown("""
                    **Time Periods:**  
                    - Weekly (5 days)  
                    - Monthly (21 days)  
                    - Yearly (252 days)  
                    Standard deviation is calculated over these time periods to assess volatility.
                    """)
                ])
            ]),
        ], width=12),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bollinger Bands"),
                dbc.CardBody([
                    dcc.Markdown("""
                    **Calculation:**  
                    Bollinger Bands are computed using a 20-day rolling average plus and minus twice the 20-day rolling standard deviation.
                    """)
                ])
            ]),
        ], width=12),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Heiken Ashi"),
                dbc.CardBody([
                    dcc.Markdown("""
                    **Formulae:**  
                    - High: High price of the day  
                    - Low: Low price of the day  
                    - Open: Average of the previous day’s open and close prices  
                    - Close: Average of the open, close, high, and low prices of the current period  
                    Heiken Ashi candles are used as individual features and combined to determine their importance in our model.
                    """)
                ])
            ]),
        ], width=12),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stochastic Oscillator"),
                dbc.CardBody([
                    dcc.Markdown("""
                    **Calculation:**  
                    Stochastic Oscillator (SO) is computed as K-D, where K and D are defined as follows:
                    - K: Closing price minus the lowest price in the last 14 days over the difference between the highest and lowest prices in the last 14 days.
                    - D: 3-day rolling average of K.
                    """)
                ])
            ]),
        ], width=12),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bull and Bear Powers"),
                dbc.CardBody([
                    dcc.Markdown("""
                    **Calculation:**  
                    - Bull Power: Difference between the high prices and the 14-day Simple Moving Average (SMA).
                    - Bear Power: Difference between the low prices and the 14-day SMA.
                    """)
                ])
            ]),
        ], width=12),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Relative Strength Index (RSI)"),
                dbc.CardBody([
                    dcc.Markdown("""
                    **Calculation:**  
                    RSI is computed using a 14-day window and is calculated as 100 - (100 / (1 + (average gain / average loss))).
                    """)
                ])
            ]),
        ], width=12),
    ]),
    dcc.ConfirmDialog(id='confirm-running', message='Strategy already running.'),
    dcc.ConfirmDialog(id='confirm-closed', message='Close all positions successfully.'),
    dbc.Row([
        dbc.Col(html.Button("Start Trading", id="start-trade", className="btn btn-success"), width=3,style={'margin-left': 'auto', 'margin-right': 'auto','margin-top': '20px'}),
        dbc.Col(html.Button("Close All Trades", id="kill-switch", className="btn btn-danger"), width=3, style={'margin-top': '20px'}),
    ], justify="start",style={'margin-top': '50px', 'border-top': '1px solid #ccc', 'box-shadow': '0 -1px 0 #ccc'}),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='strategy-selector',
            options=[
                {'label': 'Strategy 1', 'value': '1'},
                {'label': 'Strategy 2', 'value': '2'}
            ],
            value='1',  
            style={'width': '100%','margin-left': 'auto', 'margin-right': 'auto',}
        ), width=3),
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='strategy-description'), width=12),
        
    ] ,style={'margin-top': '50px'}),

    dbc.Row([
        dbc.Col(html.Div(id='current-balance', className="text-center"), width=12),
    ]),

    dbc.Row(dbc.Col(dash_table.DataTable(
        id='financial-metrics-table',
        columns=[
            {'name': 'Instrument', 'id': 'Instrument'},
            {'name': 'Target PNL', 'id': 'Target PNL'},
            {'name': 'Current PNL', 'id': 'Current PNL'},
            {'name': 'Stoploss PNL', 'id': 'Stoploss PNL'},
        ],
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': '#28a745',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_table={'overflowX': 'auto'}
    ), width=12),style={'margin-top': '30px'}),

    dbc.Row([
        dbc.Col(dcc.Graph(id='net-units-graph'), width=6),
        dbc.Col(dcc.Graph(id='unrealized-pl-graph'), width=6),
    ]),

    dbc.Row([dbc.Col(dcc.Graph(id='margin-used-graph'), width=6),
             dbc.Col(dcc.Graph(id='strategy-vs-benchmark'), width=6)]),
    # dbc.Row(dbc.Col(dcc.Graph(id='strategy-vs-benchmark'), width=6)),
], fluid=True)
#update time
@app.callback(
    Output('time-display', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_time(n):
    current_time = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    return f"Real time Dashboard: {current_time} UTC"

# update price display
last_prices = {}
@app.callback(
    Output('price-table', 'data'),
    Output('price-table', 'style_data_conditional'),
    Input('interval-component', 'n_intervals')
)
def update_price_display(n):
    instruments = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]
    prices_info = fetch_latest_prices(instruments)
    global last_prices

    styles = []
    data = []
    for info in prices_info:
        parsed_time = parser.parse(info['time'])
        formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
        
        row = {
            'time': formatted_time,
            'instrument': info['instrument'],
            'bid': info['bid'],
            'ask': info['ask'],
            'mid': info['mid']
        }
        data.append(row)
        if info['instrument'] in last_prices:
            last_info = last_prices[info['instrument']]
            if info['mid'] > last_info['mid']:
                styles.append({'if': {'filter_query': f'{{instrument}} eq "{info["instrument"]}"'}, 'color': 'green'})
            elif info['mid'] < last_info['mid']:
                styles.append({'if': {'filter_query': f'{{instrument}} eq "{info["instrument"]}"'}, 'color': 'red'})

    last_prices = {info['instrument']: info for info in prices_info}

    return data, styles

# update candlestick
@app.callback(
    Output('live-candlestick', 'figure'),
    [Input('graph-interval', 'n_intervals'),
     Input('instrument-selector', 'value')]
)
def update_graph(n, instrument_name):

    data = fetch_data_to_show(instrument_name)
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f'Real-Time Candlestick Chart for {instrument_name}',
        xaxis_rangeslider_visible=True
    )
    return fig

# app.layout.children.append(dcc.ConfirmDialog(id='confirm-running', message='Strategy already running.'))


@app.callback(
    Output('output-state', 'children'),
    Input('submit-button', 'n_clicks'),
    State('EUR_USD_risk_factor', 'value'), State('EUR_USD_risk_reward', 'value'),
    State('USD_JPY_risk_factor', 'value'), State('USD_JPY_risk_reward', 'value'),
    State('GBP_USD_risk_factor', 'value'), State('GBP_USD_risk_reward', 'value'),
    State('USD_CHF_risk_factor', 'value'), State('USD_CHF_risk_reward', 'value'),
    State('AUD_USD_risk_factor', 'value'), State('AUD_USD_risk_reward', 'value')
)
def update_risk_preferences(n_clicks, *args):
    if n_clicks > 0:
        risk_factors = {
            "EUR_USD": float(args[0]) / 100,
            "USD_JPY": float(args[2]) / 100,
            "GBP_USD": float(args[4]) / 100,
            "USD_CHF": float(args[6]) / 100,
            "AUD_USD": float(args[8]) / 100,
        }
        risk_rewards = {
            "EUR_USD": float(args[1]),
            "USD_JPY": float(args[3]),
            "GBP_USD": float(args[5]),
            "USD_CHF": float(args[7]),
            "AUD_USD": float(args[9]),
        }

        # update redis
        r.set('risk_factors', json.dumps(risk_factors))
        r.set('risk_rewards', json.dumps(risk_rewards))

        return "Risk factors and rewards updated!"
    return "Enter values and press submit."
# Callback to start trading
@app.callback(
    Output('confirm-running', 'displayed'),
    [Input('start-trade', 'n_clicks')],
    [State('strategy-selector', 'value')],
    prevent_initial_call=True  # prevent the message from showing up when the page is loaded
)
def handle_start_trade(n_clicks,selected_strategy):
    if r.get('is_trading_running').decode('utf-8') == 'True':
        # if trading is already running, do nothing but show message
        print("Trading already running")
        return True
    else:
        # if trading is not running, start trading
        r.set('strategy',selected_strategy)
        r.set('is_trading_running', 'True')  # set the flag to True
        print("Starting trading now...")
        # trade_thread = threading.Thread(target=start_trading)
        # trade_thread.start()
        return False  # do not show message

# Callback for the kill switch
@app.callback(
    Output('confirm-closed', 'displayed'),
    [Input('kill-switch', 'n_clicks')],
    prevent_initial_call=True
)
def kill_switch(n_clicks):
    # if n_clicks is None:
    #     return False
    # triggered = callback_context.triggered[0]['prop_id'].split('.')[0]
    # is_trading = r.get('is_trading_running').decode('utf-8') == 'True'
    # if triggered == 'kill-switch' and is_trading:
    #     print("CLICK TRY: Closing all trades")
    #     close_all_trades(client, accountID)
    #     r.set('is_trading_running', 'False')
    #     # for instrument in instruments:
    #     #     r.hset(instrument, 'inposition', 'False')
    #     return True
    # elif not is_trading:
    #     return False
    # else:
    #     return False if is_trading else  True
    if n_clicks is None:
        return False
    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    is_trading = r.get('is_trading_running').decode('utf-8') == 'True'

    if button_id == 'kill-switch' and is_trading:
        close_all_trades(client, accountID)
        r.set('is_trading_running', 'False')
        return True  
    return False  

# define callback to update strategy description
@app.callback(
    Output('strategy-description', 'children'),
    Input('strategy-selector', 'value')
)
def update_strategy_description(selected_strategy):
    if selected_strategy == '1':
        return "Description of Strategy 1: Binary signal based on the predicted value. If the predicted value is positive (negative), we enter (exit) a position. This strategy places less importance on the accuracy of our predictions."
    elif selected_strategy == '2':
        return "Description of Strategy 2: Utilise the standard deviation of the difference between the predicted and actual values as a confidence band. We trade only if the predicted value is higher or lower than the band, indicating that the predicted value is statistically different from zero."
    else:
        return "Please select a trading strategy."
    
# define callback to update current balance
@app.callback(
    Output('current-balance', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_current_balance(n):
    equity = get_account_equity()
    balance = get_current_balance()
    return html.H1(f"Current equity: {equity:.2f}, Current balance: {balance:.2f}", style={'position': 'fixed', 'top': 0 ,'color': 'green'})
    # return html.H5(f"Current equity: {equity:.2f}, Current balance: {balance:.2f}")

# # define callback to update target profit and stop loss
@app.callback(
    Output('financial-metrics-table', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_financial_metrics(n):
    data = load_data()  
    return data
    # return data.to_dict('records')  

# define callback to update open positions
instruments = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD"]
@app.callback(
    [Output('net-units-graph', 'figure'),  
     Output('unrealized-pl-graph', 'figure'),
     Output('margin-used-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_open_positions(n):
    positions = get_open_positions()
    # data = []
    net_units_data = []
    unrealized_pl_data = []
    margin_used_data = []
    for instrument in instruments:
        
        position = next((p for p in positions if p['instrument'] == instrument), None)
        if not position:
            # long_units = short_units = 0  
            net_units = 0 #default to 0 if no position
            unrealized_pl = 0
            margin_used = 0
        else:
            # long_units = float(position['long']['units'])
            # short_units = float(position['short']['units'])
            net_units = float(position['long']['units']) + float(position['short']['units'])
            unrealized_pl = float(position['unrealizedPL'])
            margin_used = float(position['marginUsed'])
        # color = 'green' if net_units >= 0 else 'red'
    

        # data.append(go.Bar(name=instrument, x=[instrument], y=[net_units], marker=dict(color=color)))

        net_units_data.append(go.Bar(name=instrument, x=[instrument], y=[net_units], marker=dict(color='blue' if net_units >= 0 else 'orange')))
        unrealized_pl_data.append(go.Bar(name=instrument, x=[instrument], y=[unrealized_pl], marker=dict(color='green' if unrealized_pl >= 0 else 'red')))
        margin_used_data.append(go.Bar(name=instrument, x=[instrument], y=[margin_used], marker=dict(color='purple')))

    layout_units = go.Layout(title="Net Units Overview", yaxis=dict(title='Net Units'))
    layout_pl = go.Layout(title="Unrealized P&L Overview", yaxis=dict(title='Unrealized P&L'))
    layout_margin = go.Layout(title="Margin Used Overview", yaxis=dict(title='Margin Used'))

    fig_units = go.Figure(data=net_units_data, layout=layout_units)
    fig_unrealized_pl = go.Figure(data=unrealized_pl_data, layout=layout_pl)
    fig_margin_used = go.Figure(data=margin_used_data, layout=layout_margin)
    return fig_units, fig_unrealized_pl, fig_margin_used






start_time = datetime.datetime.now()
last_equity = get_account_equity()
last_price = get_current_price("EUR_USD")

equity_changes = []
price_changes = []
time_stamps = [start_time]

@app.callback(
    Output('strategy-vs-benchmark', 'figure'),
    Input('graph-interval', 'n_intervals')
)
def update_performance_graph(n):
    global last_equity, last_price, start_time
    current_equity = get_account_equity()
    current_price = get_current_price("EUR_USD")
    current_time = datetime.datetime.now()
    
    equity_change_pct = ((current_equity - last_equity) / last_equity) * 100 if last_equity != 0 else 0
    price_change_pct = ((current_price - last_price) / last_price) * 100 if last_price != 0 else 0
    time_stamps.append(current_time)

    last_equity = current_equity
    last_price = current_price
    
    equity_changes.append(equity_change_pct)
    price_changes.append(price_change_pct)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_stamps, y=equity_changes, mode='lines+markers', name='Equity % Change'))
    fig.add_trace(go.Scatter(x=time_stamps, y=price_changes, mode='lines+markers', name='EUR/USD % Change'))

    fig.update_layout(title='Real-Time Performance vs EUR/USD',
                      xaxis_title='Time',
                      yaxis_title='Percentage Change (%)',
                      legend_title='Legend')
    
    return fig






if __name__ == '__main__':
    global r
    global client
    global accountID 

    r = redis.Redis(host='localhost', port=6379, db=0)

    accountID, client = login()
    app.run_server(debug=True)
