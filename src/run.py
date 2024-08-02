#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import requests
from strategy.strategy import Strategy
from strategy.moving_average import MovingAverage
from strategy.z_score import ZScore
from strategy.rsi import RSI
from strategy.roc import ROC
from strategy.percentile import Percentile
from strategy.min_max import MinMax
from strategy.robust import Robust
from joblib import Parallel, delayed
from optimization import Optimization
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

load_dotenv()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('mode.string_storage', 'pyarrow')

# Paramenter
COMMISSION = 0.0005
GLASSNODE_API_KEY = os.getenv('GLASSNODE_API_KEY')
ASSET = 'BTC'
INTERVAL = '10m'
WINDOW_SIZE_PERRCENT = 0.10
NUM_WINDOW_SIZES = 40

# Glassnode fetching BTC price
def fetch_asset_price(asset, interval, api_key):
    url = "https://api.glassnode.com/v1/metrics/market/price_usd_close"
    params = {
        'a': asset,
		's': '1577836800', # since 2020-01-01 UTC
        'i': interval,
        'api_key': api_key,
		'timestamp_format': 'unix'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        #df = df.convert_dtypes(dtype_backend='pyarrow')
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Data
btc_price_df = fetch_asset_price(ASSET, INTERVAL, GLASSNODE_API_KEY)
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])

# Factor
directory = '../data/factors'
price_factor_dict = {}
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        #factor_df = pd.read_csv(file_path)[['timestamp', 'value']]
        factor_df = pd.read_csv(file_path)
        #factor_df = factor_df.convert_dtypes(dtype_backend='pyarrow')
        factor_df.columns = ['Date', 'Target']
        factor_df["Date"] = pd.to_datetime(factor_df["Date"])
        factor_df['Target'] = factor_df['Target'].shift(1) # shift data to avoid bias
        price_factor_df = pd.merge(btc_price_df, factor_df, how='inner', on='Date')

        price_factor_dict[filename] = price_factor_df

threshold_params = {
        'ZScore': np.round(np.linspace(-3, 3, 42), 3),
        'MovingAverage': np.round(np.linspace(-5, 5, 32), 3),
        'RSI': np.round(np.linspace(0.2, 0.8, 32), 4),
        'ROC': np.round(np.linspace(-1, 1, 32), 4),
        'MinMax': np.round(np.linspace(0.1, 0.9, 42), 4),
        'Robust': np.round(np.linspace(0, 5, 32), 4),
        'Percentile': np.round(np.linspace(0.1, 0.9, 32), 4)
    }
strategy_list = ['ZScore', 'MovingAverage', 'RSI', 'ROC', 'MinMax', 'Robust', 'Percentile']
long_short_params = ['long', 'short', 'both']
condition_params = ['lower', 'higher']

running_list = []
for filename in price_factor_dict:
    for strategy in strategy_list:
        for long_short in long_short_params:
            for condition in condition_params:
                test = {
                        'Metric': filename,
                        'Strategy': strategy,
                        'Strategy Type': long_short,
                        'Condition': condition
                    }
                running_list.append(test)



overall_result = pd.DataFrame()

def running_single_strategy(strategy, metric, price_factor_df, long_short, condition):

    Strategy.bps = 7
    window_size_list = np.linspace(2, int(len(price_factor_df) * WINDOW_SIZE_PERRCENT),
                                   NUM_WINDOW_SIZES, dtype=int)

    test = Optimization(strategy, price_factor_df, window_size_list, threshold_params[strategy], target="Target", price='Price', long_short=long_short, condition=condition)
    test.run()


    test_output = {
            'Metric': metric,
            'Output': test.get_best()
            }
    return test_output


if __name__ == "__main__":

    results_list = []
    with tqdm_joblib(tqdm(leave=False, desc="Processing", total=len(running_list))) as progress_bar:
        parallel_results =  Parallel(n_jobs=4, timeout=600)(delayed(running_single_strategy)(run['Strategy'], run['Metric'], price_factor_dict[run['Metric']], run['Strategy Type'], run['Condition'])for run in running_list)

    for result in parallel_results:
        ouput = result['Output']

        results_list.append({ "Metric": result['Metric'] } | ouput.dump_data())


    results_list_df = pd.DataFrame(results_list)
    results_list_df.to_csv("btc_factor_test2.csv")
    print(results_list_df)

    print('done')
    #print(result.calmar)
    #print(result.sharpe)
    #result.result_df.to_csv("btc_liq_test.csv")
    #print(result.result_df)
    #result.plot_graph()

