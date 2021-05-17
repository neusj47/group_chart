# -*- coding: utf-8 -*-

# 특정 섹터와 종목 수익률, DrawDown(기간 낙폭) 비교할 수 있는 함수
# 관심 섹터와, TICKER를 입력합니다.
# 섹터를 입력합니다. (1차)
# 섹터에 속한 TICKER 리스트가 보여집니다. (2차)
# 섹터별 TICKER의 수익률, Drawdown 데이터를 호출 후 Graph를 산출합니다. (3차)

import dash  # Dash 1.16 or higher
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from datetime import date
import datetime
import pandas_datareader.data as web
import dash_bootstrap_components as dbc

# need to pip install statsmodels for trendline='ols' in scatter plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# TICKER를 입력합니다.
TICKER = ['INTC','NVDA','QCOM','GOOGL','FB']

start = date(2016, 1, 1)
end = datetime.datetime.now()

# 수익률, 거래량 데이터를 산출합니다.
dfs = web.DataReader(TICKER[0], 'yahoo', start, end)
dfs.reset_index(inplace=True)
dfs.set_index("Date", inplace=True)
dfs['Return'] = (dfs['Close'] / dfs['Close'].shift(1)) - 1
dfs['Return(cum)'] = (1 + dfs['Return']).cumprod()
dfs = dfs.dropna()
dfs.loc[:,'TICKER'] = TICKER[0]
dfs['DD'] = dfs['Close']/(dfs['Close'].rolling(252, min_periods=1).max()) # 252영업일 중 가장 높은 종가 대비 현재가격 (기간 낙폭)
dfs['MDD'] = dfs['DD'].rolling(252, min_periods=1).min() # 252영업일 동안의 기간 낙폭 중 가장 낮은 값
df = dfs

for i in range(1,len(TICKER)):
    start = date(2016, 1, 1)
    end = datetime.datetime.now()
    dfs = web.DataReader(TICKER[i], 'yahoo', start, end)
    dfs.reset_index(inplace=True)
    dfs.set_index("Date", inplace=True)
    dfs['Return'] = (dfs['Close'] / dfs['Close'].shift(1)) - 1
    dfs['Return(cum)'] = (1 + dfs['Return']).cumprod()
    dfs = dfs.dropna()
    dfs.loc[:,'TICKER'] = TICKER[i]
    dfs['DD'] = dfs['Close'] / (dfs['Close'].rolling(252, min_periods=1).max())
    dfs['MDD'] = dfs['DD'].rolling(252, min_periods=1).min()
    df = df.append(dfs)

df_A = df[(df['TICKER'] == 'INTC')|(df['TICKER'] == 'NVDA')|(df['TICKER'] == 'QCOM')]
df_B = df[(df['TICKER'] == 'GOOGL')|(df['TICKER'] == 'FB')]
df_A.insert(9,'Sector','Telecom_Service')
df_B.insert(9,'Sector','IT')

dff = pd.concat([df_A, df_B])

# 데이터타입(Date)변환 문제로 csv 저장 후, 다시 불러옵니다. (파일 경로 설정 필요!!)
dff = dff.reset_index().rename(columns={"index": "id"})
dff.to_csv('pricevolume.csv', index=False, encoding='cp949')
dff = pd.read_csv('C:/Users/ysj/PycharmProjects/group_chart/pricevolume.csv')


app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.Label("Sector:", style={'fontSize': 20, 'textAlign': 'center'}),
            dcc.Dropdown(
                id='dropdown_sector',
                options=[{'label': s, 'value': s} for s in sorted(dff.Sector.unique())],
                value='IT',
                clearable=False)
            ], xs=12, sm = 12, md = 12, lg =5, xl=5),

        dbc.Col([
            html.Label("TICKER:", style={'fontSize': 20, 'textAlign': 'center'}),
            dcc.Dropdown(id='dropdown-TICKER',options=[],multi=True)
            ], xs=12, sm = 12, md = 12, lg =5, xl=5)
        ],  no_gutters=False , justify='start')
        , dcc.Graph(id='display-map', figure={})
        , dcc.Graph(id='display-map2', figure={})

], fluid= True)


# 선택된 Sector의 Ticker가 보여지도록 합니다.(callback 1차)
@app.callback(
    Output('dropdown-TICKER', 'options'),
    Input('dropdown_sector', 'value')
)
def set_cities_options(chosen_Sector):
    dfs = dff[dff.Sector==chosen_Sector]
    return [{'label': c, 'value': c} for c in sorted(dfs.TICKER.unique())]

@app.callback(
    Output('dropdown-TICKER', 'value'),
    Input('dropdown-TICKER', 'options')
)

# 선택된 섹터의 TICKER 정보가 호출됩니다. (call back 2차)
def set_cities_value(available_options):
    return [x['value'] for x in available_options]

# 섹터와 TICKER 수익률을 바탕으로 그래프를 산출합니다. (callback 3차)
@app.callback(
    Output('display-map', 'figure'),
    Input('dropdown-TICKER', 'value'),
    Input('dropdown_sector', 'value')
)
def update_grpah(selected_TICKER, selected_Sector):
    if len(selected_TICKER) == 0:
        return dash.no_update
    else:
        dfs = dff[(dff.Sector==selected_Sector) & (dff.TICKER.isin(selected_TICKER))]

        fig = px.scatter(dfs, x='Date', y='Return(cum)',
                         color = 'TICKER',
                         hover_name='TICKER',
                         title = 'CumRtn for TICKER'
                         )
        return fig


# 섹터와 TICKER 수익률을 바탕으로 그래프를 산출합니다. (callback 3차)
@app.callback(
    Output('display-map2', 'figure'),
    Input('dropdown-TICKER', 'value'),
    Input('dropdown_sector', 'value')
)
def update_grpah(selected_TICKER, selected_Sector):
    if len(selected_TICKER) == 0:
        return dash.no_update
    else:
        dfs = dff[(dff.Sector==selected_Sector) & (dff.TICKER.isin(selected_TICKER))]

        fig = px.line(dfs, x='Date', y='DD',
                         color = 'TICKER',
                         hover_name='TICKER',
                         title='DrawDown for TICKER'
                         )
        return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8005)

